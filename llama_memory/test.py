import argparse
from os import path
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import CrossEntropyLoss
from torcheval.metrics.text import Perplexity

from datasets import MemoryDataset

from llama import ModelArgs, Transformer, Tokenizer
from trainer import MEM_LLAMA_CONFIG


@torch.inference_mode()
def eval_all(model, data, method, device, name, wandb=None):
    metric = Perplexity(ignore_index=0, device=device)
    metric.compute()
    for step, (src, tgt, mem) in enumerate(data):
        if method == "prepend":
            mem[-1] = 0  # remove eos token
            src = torch.cat((mem, src), dim=1)
            tgt = torch.cat((mem, tgt), dim=1)
        if method != "encoder":
            mem = None
        else:
            mem = mem.to(device)

        _, to_compare = model(src.to(device), 0, mem)
        metric.update(
            to_compare.transpose(1, 2),
            tgt.to(device)
        )

        if wandb is not None:
            wandb.log({
                f"{name}/Step": step,
                f"{name}/Perplexity": metric.compute()
            })
    return metric.compute()


def main():
    parser = argparse.ArgumentParser(description="Run LLaMA")
    parser.add_argument("--checkpoint-dir",
                        dest="CKPT_DIR",
                        default="./checkpoints",
                        help="Location of the checkpoints directory")
    parser.add_argument("--data-path",
                        default="./data",
                        help="Location of the data directory")
    parser.add_argument("--model",
                        dest="MODEL",
                        default="7B",
                        help="The model to use. Currently only supports 7B")
    parser.add_argument("--seq-len",
                        dest="SEQ_LEN",
                        type=int,
                        default=512,
                        help="Maximum sequence length (up to 2048)")
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="Maximum batch size")
    parser.add_argument("--method",
                        type=str,
                        choices=["encoder", "prepend", "no_mem"],
                        required=True,
                        help="The method to use for memory injection")
    parser.add_argument("--device",
                        type=str,
                        choices=["cpu", "cuda"],
                        default="cuda",
                        help="The device to use.")
    parser.add_argument("--wandb",
                        action="store_true",
                        help="Use W&B")
    args = parser.parse_args()
    device = args.device

    config = MEM_LLAMA_CONFIG
    decoder_path = path.join(args.CKPT_DIR, args.MODEL, "consolidated.00.pth")
    tokenizer_path = path.join(args.CKPT_DIR, "tokenizer.model")
    decoder = torch.load(decoder_path, map_location="cpu")
    if args.method == "encoder":
        encoder_path = path.join(args.CKPT_DIR, args.MODEL, "encoder.pth")
        encoder = torch.load(encoder_path, map_location="cpu")
        config = encoder["config"]
    with open(path.join(args.CKPT_DIR, args.MODEL, "params.json"), "r") as f:
        params = json.loads(f.read())
    tokenizer = Tokenizer(model_path=tokenizer_path)
    params["vocab_size"] = tokenizer.n_words

    dec_seq_len = config["dec_seq_len"]
    if args.method == "prepend":
        dec_seq_len += config["enc_seq_len"]  # need more seq_len to prepend

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(ModelArgs(
        max_dec_seq_len=dec_seq_len,
        max_enc_seq_len=config["enc_seq_len"],
        max_batch_size=min(args.batch_size, config["batch_size"]),
        decoder_memory_start=config["decoder_memory_start"],
        decoder_memory_end=config["decoder_memory_end"],
        encoder_n_layers=config["enc_n_layers"],
        **params
    ))
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loading decoder checkpoint from path {decoder_path}")
    model.load_state_dict(decoder, strict=False)

    if args.method == "encoder":
        print(f"Loading encoder checkpoint from path {encoder_path}")
        model.encoder.load_state_dict(encoder["encoder_state_dict"], strict=False)
        model.enc_tok_embeddings.load_state_dict(encoder["encoder_token_embeddings"], strict=False)

    DATASETS_NAMES = ["curio", "dream", "sgd", "wow"]

    if args.wandb:
        import wandb

        wandb.init(
            project="memory-LLaMA",
            name="Testing: " + args.method,
            config=config
        )

    data_test = []
    data_eval = []
    for dataset_name in DATASETS_NAMES:
        data_test.append(
            MemoryDataset(path.join(args.data_path, dataset_name),
                          tokenizer,
                          config["dec_seq_len"],
                          config["enc_seq_len"],
                          "test")
        )
        data_eval.append(
            MemoryDataset(path.join(args.data_path, dataset_name),
                          tokenizer,
                          config["dec_seq_len"],
                          config["enc_seq_len"],
                          "val")
        )

    data_test = ConcatDataset(data_test)
    data_eval = ConcatDataset(data_eval)
    data_test = DataLoader(
        data_test,
        batch_size=min(args.batch_size, config["batch_size"]),
        shuffle=True
    )
    data_eval = DataLoader(
        data_eval,
        batch_size=min(args.batch_size, config["batch_size"]),
        shuffle=True
    )
    model = model.to(device)

    print(eval_all(model, data_eval, args.method, device, name="Val", wandb=wandb if args.wandb else None))
    print(eval_all(model, data_test, args.method, device, name="Test", wandb=wandb if args.wandb else None))


if __name__ == "__main__":
    main()
