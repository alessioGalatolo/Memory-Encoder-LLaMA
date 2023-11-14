import json
from os import path, remove, rmdir, rename
import subprocess
from glob import glob


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")

# requires parlai (pip install parlai)
subprocess.run(["parlai", "display_data", "--task", "wizard_of_wikipedia", "--dp", current_dir])
for file in glob(path.join(current_dir, "wizard_of_wikipedia", "*")):
    rename(file, path.join(current_dir, file.split("/")[-1]))
print("Dataset downloaded")

print("Parsing files...")
for split, filename in zip(["train", "val", "test"], ["train", "valid_topic_split", "test_topic_split"]):
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, f"{filename}.json"), "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
        for dialogs in data:
            msgs2write = []
            mem_data = []
            persona = dialogs["persona"]
            mem_data.append(persona)
            for dialog in dialogs["dialog"]:
                text = dialog["text"]
                if "wizard" in dialog["speaker"].lower():
                    try:
                        mem_data.append(list(dialog["checked_sentence"].values())[0])
                    except IndexError:
                        mem_data.append(list(dialog["checked_passage"].values())[0])
                    for passage in dialog["retrieved_passages"]:
                        mem_data.extend(list(passage.values())[0])
                    mem.write(" [SEP] ".join(mem_data) + "\n")
                    src.write(" [SEP] ".join(msgs2write) + "\n")
                    tgt.write(text + "\n")
                msgs2write.append(text)
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
remove(path.join(current_dir, "wizard_of_wikipedia", ".built"))
rmdir(path.join(current_dir, "wizard_of_wikipedia"))
print("Done!")
