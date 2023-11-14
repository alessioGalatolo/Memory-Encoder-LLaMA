from os import path
import torch
from torch.utils.data import Dataset


# generic memory dataset
class MemoryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_tokens_dec, max_tokens_enc, split):
        assert split in ["train", "val", "test"]

        self.src_file = open(path.join(data_path, f"{split}.src"), "r").readlines()
        self.tgt_file = open(path.join(data_path, f"{split}.tgt"), "r").readlines()
        self.mem_file = open(path.join(data_path, f"{split}.mem"), "r").readlines()

        assert len(self.src_file) == len(self.tgt_file) and len(self.tgt_file) == len(self.mem_file)

        self.max_tokens_dec = max_tokens_dec
        self.max_tokens_enc = max_tokens_enc
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_file)

    def __getitem__(self, index):
        src = self.src_file[index]
        tgt = self.tgt_file[index]
        mem = self.mem_file[index]

        tgt = src + " " + tgt
        src = torch.tensor(self.tokenizer.encode(src, bos=True, eos=False), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(tgt, bos=True, eos=True), dtype=src.dtype)
        mem = torch.tensor(self.tokenizer.encode(mem, bos=True, eos=True), dtype=src.dtype)

        padding_src = self.max_tokens_dec - src.shape[0]
        if padding_src > 0:
            src = torch.cat((src, torch.zeros(padding_src, dtype=tgt.dtype)))
        else:
            src = src[: self.max_tokens_dec]

        padding_dec = self.max_tokens_dec - tgt.shape[0]
        if padding_dec > 0:
            tgt = torch.cat((tgt, torch.zeros(padding_dec, dtype=tgt.dtype)))
        else:
            tgt = tgt[: self.max_tokens_dec]

        padding_enc = self.max_tokens_enc - mem.size(0)
        if padding_enc > 0:
            mem = torch.cat((mem, torch.zeros(padding_enc, dtype=mem.dtype)))
        else:
            mem = mem[: self.max_tokens_enc]

        return src, tgt, mem
