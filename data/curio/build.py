import urllib.request
import json
from os import path, remove
from collections import defaultdict
from glob import glob


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/fact_db_links.json", path.join(current_dir, "facts.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.train.json", path.join(current_dir, "train.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.val.json", path.join(current_dir, "val.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.test.json", path.join(current_dir, "test.json"))
print("Dataset downloaded")

with open(path.join(current_dir, "facts.json"), "r") as raw_file:
    facts = json.load(raw_file)

facts = defaultdict(lambda: {"text": ""}, facts["linked_facts"])

print("Parsing files...")
for split in ["train", "val", "test"]:
    with open(path.join(current_dir, f"{split}.json"), "r", encoding="utf-8") as raw_file:
        data = json.load(raw_file)
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        for dialog in data["dialogs"]:
            msgs = dialog["messages"]
            msgs2write = []
            tgt2write = ""
            for msg in msgs:
                if msg["sender"] == "assistant":
                    facts2write = list(filter(None, map(lambda x: facts[str(x["fid"])]["text"], msg["facts"])))
                    mem.write(" [SEP] ".join(facts2write) + "\n")
                    src.write(" [SEP] ".join(msgs2write) + "\n")
                    tgt.write(msg["message"] + "\n")
                msgs2write.append(msg["message"])
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
print("Done!")
