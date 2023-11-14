import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
import json
from os import makedirs, path, environ
from shutil import rmtree

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datasets import MemoryDataset
from llama import Tokenizer, Transformer, ModelArgs


MEM_LLAMA_CONFIG = {
        "model": "7B",
        "lr": 1e-4,
        "batch_size": 32,
        "warmup_epochs": 2,
        "epochs": 5,
        "dec_seq_len": 512,
        "enc_seq_len": 512,
        "label_smoothing": 0.0,
        "decoder_memory_start": 0,
        "decoder_memory_end": 32,
        "enc_n_layers": 8
    }


def train(
    model_path,
    data_path,
    output,
    resume,
    use_wandb,
    debug,
    distributed_training,
    config=MEM_LLAMA_CONFIG
):
    DATASETS_NAMES = ["curio", "dream", "sgd", "wow"]
    if distributed_training:
        init_process_group(backend="nccl")

    if resume:
        resume_dict = torch.load(resume, map_location='cpu')
        config = resume_dict["config"]
        print("Resuming previous run, using config:")
        print(config)

    if use_wandb:
        import wandb

        wandb.init(
            project="memory-LLaMA",
            name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
            config=config,
            resume=bool(resume),
            group="DDP" if distributed_training else None
        )
        config = wandb.config
        wandb.define_metric("Train/Step")
        wandb.define_metric("Train/*", step_metric="Train/Step")
        wandb.define_metric("Val/Step")
        wandb.define_metric("Val/*", step_metric="Val/Step")
        train_step = 0
        val_step = 0

    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in environ:
        device = int(environ["LOCAL_RANK"])

    tokenizer = Tokenizer(path.join(model_path, "tokenizer.model"))
    data_train = []
    data_eval = []
    for dataset_name in DATASETS_NAMES:
        data_train.append(
            MemoryDataset(path.join(data_path, dataset_name),
                          tokenizer,
                          config["dec_seq_len"],
                          config["enc_seq_len"],
                          "train")
        )
        data_eval.append(
            MemoryDataset(path.join(data_path, dataset_name),
                          tokenizer,
                          config["dec_seq_len"],
                          config["enc_seq_len"],
                          "val")
        )

    data_train = ConcatDataset(data_train)
    data_eval = ConcatDataset(data_eval)
    data_train = DataLoader(
        data_train,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(data_train, shuffle=True) if distributed_training else None,
        shuffle=True if not distributed_training else False
    )
    data_eval = DataLoader(
        data_eval,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(data_eval, shuffle=True) if distributed_training else None,
        shuffle=True if not distributed_training else False
    )

    with open(path.join(model_path, config["model"], "params.json"), "r") as f:
        params = json.loads(f.read())
    params["vocab_size"] = tokenizer.n_words
    model = Transformer(ModelArgs(
        max_dec_seq_len=config["dec_seq_len"],
        max_enc_seq_len=config["enc_seq_len"],
        max_batch_size=config["batch_size"],
        decoder_memory_start=config["decoder_memory_start"],
        decoder_memory_end=config["decoder_memory_end"],
        encoder_n_layers=config["enc_n_layers"],
        **params
    ))
    checkpoint = torch.load(path.join(model_path, config["model"], "consolidated.00.pth"), map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)

    if use_wandb:
        wandb.watch(model, log_freq=100)

    for name, param in model.named_parameters():
        if "enc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer = torch.optim.AdamW(
        params=chain(
            model.encoder.parameters(),
            model.enc_tok_embeddings.parameters()
        ),
        lr=config["lr"],
        betas=(0.9, 0.95)
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        (config["epochs"] - config["warmup_epochs"]) * len(data_train),
        eta_min=config["lr"] * 0.1
    )
    warmup_lr_scheduler = LinearLR(
        optimizer,
        0.1,
        1,
        config["warmup_epochs"] * len(data_train) / 2
    )
    scaler = GradScaler()

    epoch_0 = 0
    if resume:
        print("Resuming previous run, loading state dicts")
        scaler.load_state_dict(resume_dict["scaler"])
        optimizer.load_state_dict(resume_dict["optimizer"])
        lr_scheduler.load_state_dict(resume_dict["lr_scheduler"])
        warmup_lr_scheduler.load_state_dict(resume_dict["warmup_lr_scheduler"])
        model.encoder.load_state_dict(resume["encoder_state_dict"])
        model.enc_tok_embeddings.load_state_dict(resume["encoder_token_embeddings"])
        epoch_0 = resume_dict["epoch"]
        print("Loaded and ready!")

    if distributed_training:
        model = DDP(model, device_ids=[device])

    last_checkpoints = []  # used to delete old checkpoints while run is continuing
    loss_fn = CrossEntropyLoss(ignore_index=0, label_smoothing=config["label_smoothing"])
    for epoch in range(epoch_0, config["epochs"]):

        model.train()
        for step, (src, tgt, mem) in enumerate(data_train):
            if epoch < config["warmup_epochs"]:
                src = torch.zeros(src.size(), dtype=src.dtype)
                tgt = mem[: config["dec_seq_len"]]
            optimizer.zero_grad()
            with autocast(
                device_type="cpu" if device == "cpu" else "cuda",
                dtype=torch.bfloat16 if device == "cpu" else torch.half
            ):
                _, to_compare = model(src.to(device), 0, mem[:, :config["enc_seq_len"]].to(device))
                loss = loss_fn(to_compare, tgt.to(device))

            if torch.isnan(loss) or torch.isinf(loss):  # TODO: remove, already checked by the scaler
                print(f"Loss is {loss.item()}, skipping batch")
                continue

            if debug:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            else:  # main case
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
            if use_wandb:
                wandb.log({
                    "Epoch": epoch + step / len(data_train),
                    "Train/Step": train_step,
                    "Train/Loss": loss.detach().cpu(),
                    "Train/LR": optimizer.state_dict()["param_groups"][0]["lr"]
                })
                train_step += 1
            if epoch < config["warmup_epochs"]:
                warmup_lr_scheduler.step()
            else:  # lr is changed with first non-warmup epoch
                lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for src, tgt, mem in data_eval:
                if epoch < config["warmup_epochs"]:
                    src = torch.zeros(src.size(), dtype=src.dtype)
                    tgt = mem[: config["dec_seq_len"]]
                with autocast(
                    device_type="cpu" if device == "cpu" else "cuda",
                    dtype=torch.bfloat16 if device == "cpu" else torch.half
                ):
                    _, to_compare = model(src.to(device), 0, mem[:, :config["enc_seq_len"]].to(device))
                    loss = loss_fn(to_compare, tgt.to(device))

                if use_wandb:
                    wandb.log({
                        "Epoch": epoch + 1,
                        "Val/Step": val_step,
                        "Val/Loss": loss.detach().cpu()
                    })
                    val_step += 1

        # save model
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
        model2save = model.module if distributed_training else model
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "warmup_lr_scheduler": warmup_lr_scheduler.state_dict(),
            "encoder_state_dict": model2save.encoder.state_dict(),
            "encoder_token_embeddings": model2save.enc_tok_embeddings.state_dict(),
            "config": dict(config)
        }

        if output is not None and (not distributed_training or device == 0):
            path_no_filename = path.join(output, now)
            makedirs(path_no_filename)
            last_checkpoints.append(path_no_filename)

            with open(path.join(path_no_filename,  "checkpoint.pth"), 'wb') as f:
                torch.save(save_dict, f)

            if len(last_checkpoints) > 3:
                rmtree(last_checkpoints.pop(0))
    if use_wandb:
        wandb.finish()
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA")
    parser.add_argument("--model-path",
                        default="./checkpoints",
                        help="Location of the checkpoints directory")
    parser.add_argument("--data-path",
                        default="./data",
                        help="Location of the data directory")
    parser.add_argument("--model",
                        choices=["7B"],
                        help="The model to use. Currently only supports 7B")
    parser.add_argument("--dec-seq-len",
                        type=int,
                        help="Maximum sequence length (up to 2048)")
    parser.add_argument("--enc-seq-len",
                        type=int,
                        help="Maximum sequence length")
    parser.add_argument("--learning-rate", "--lr",
                        type=float,
                        dest="lr",
                        help="The maximum learning rate")
    parser.add_argument("--batch-size",
                        type=int,
                        help="Maximum batch size")
    parser.add_argument("--epochs",
                        type=int,
                        help="Number of epochs")
    parser.add_argument("--output", "-o",
                        help="Checkpoints output dir")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Resume previous run, put checkpoint path here")
    parser.add_argument("--distributed",
                        action="store_true",
                        help="Distributed training")
    parser.add_argument("--wandb",
                        action="store_true",
                        help="Use W&B")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug mode, use CPU")
    args = parser.parse_args()

    config = MEM_LLAMA_CONFIG
    args_dict = args.__dict__
    for arg in args_dict:
        if arg.lower() in MEM_LLAMA_CONFIG and args_dict[arg] is not None:
            config[arg] = args_dict[arg]

    train(
        args.model_path,
        args.data_path,
        args.output,
        args.resume,
        args.wandb,
        args.debug,
        args.distributed,
        config
    )


if __name__ == "__main__":
    main()
