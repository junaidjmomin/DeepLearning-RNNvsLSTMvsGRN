import json
from pathlib import  Path 
import re 
import random 

#Variables initilization!!
datapath=Path("data/raw/dailydialog/data/dialogues.json")
OUTPUT_PATH = Path("data/processed/sequences.json")
utterances=[]

context_size=3
samples=[]


# Read THE DATASET !!
with open(datapath,"r",encoding='utf-8') as file:
    dialogues=json.load(file)

#Separating  the dataset into training and testing and validation data!!

random.seed(42)
random.shuffle(dialogues)
n = len(dialogues)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_dialogues = dialogues[:train_end]
val_dialogues = dialogues[train_end:val_end]
test_dialogues = dialogues[val_end:]



print("Total Dialogues in the chat!!",len(dialogues))
def create_sample(dialogues):
    samples = []

    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            text = turn["utterance"].lower().strip()
            text = re.sub(r"\s+", " ", text)

            if text:
                words = text.split()

                for i in range(context_size, len(words)):
                    context = words[i-context_size:i]
                    target = words[i]

                    samples.append({
                        "input": " ".join(context),
                        "target": target
                    })

    return samples

datasets = {
    "train.json": create_sample(train_dialogues),
    "validation.json": create_sample(val_dialogues),
    "test.json": create_sample(test_dialogues)
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

for filename, data in datasets.items():
    with open(OUTPUT_PATH.parent / filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(filename, ":", len(data), "samples")

    for sample in data[:3]:
        print(sample)