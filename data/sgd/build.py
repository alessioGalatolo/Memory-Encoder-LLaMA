import urllib.request
import json
from os import path, remove
from collections import defaultdict
from glob import glob


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
train_list = [f"00{i}" for i in range(1, 10)]
train_list.extend([f"0{i}" for i in range(10, 100)])
train_list.extend([str(i) for i in range(100, 128)])
val_list = train_list[:20]
test_list = val_list
for i in train_list:
    urllib.request.urlretrieve(f"https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/raw/master/train/dialogues_{i}.json", path.join(current_dir, f"train_{i}.json"))
for i in val_list:
    urllib.request.urlretrieve(f"https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/raw/master/dev/dialogues_{i}.json", path.join(current_dir, f"val_{i}.json"))
for i in test_list:
    urllib.request.urlretrieve(f"https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/raw/master/test/dialogues_{i}.json", path.join(current_dir, f"test_{i}.json"))
print("Dataset downloaded")

print("Parsing files...")
for split, split_list in zip(["train", "val", "test"], [train_list, val_list, test_list]):
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        for i in split_list:
            with open(path.join(current_dir, f"{split}_{i}.json"), "r", encoding="utf-8") as raw_file:
                data = json.load(raw_file)
            for dialog in data:
                msgs2write = []
                mem_data = []
                for msg in dialog["turns"]:
                    text = msg["utterance"]
                    if msg["speaker"] == "SYSTEM" and "service_results" in msg["frames"][0] and msg["frames"][0]["service_results"]:
                        for result in msg["frames"][0]["service_results"]:
                            mem_data.append(", ".join(map(lambda x: f"{x[0]}: {x[1]}", result.items())))
                        mem.write(" [SEP] ".join(mem_data) + "\n")
                        src.write(" [SEP] ".join(msgs2write) + "\n")
                        tgt.write(text.replace("\n", "") + "\n")
                    msgs2write.append(text.replace("\n", ""))
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
print("Done!")
