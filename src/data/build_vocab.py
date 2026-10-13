import json
from pathlib import Path

# PATHS
train_path=Path("data/processed/train.json")
vocab_path=Path("data/processed/vocab.json")

# LOADING THE DATA FROM THE FILE !
with open(train_path,"r",encoding='utf-8') as file:
    trainingdata=json.load(file)

# Store All unique words !!
unique_words=set()

for data in trainingdata:
    seprated_input_word=data["input"].split()

    # Storing all targeted unique words in the set!!
    if(seprated_input_word):
        unique_words.update(seprated_input_word)

    # Storing all the input words in the set!!
    seprated_target_word=data["target"].split()

    if(seprated_target_word):
        unique_words.update(seprated_target_word) #update method takes an iterable and add separate elements to the set!!


#Dictionary Loop!!
vocab_data={}

for ind,data in enumerate(sorted(unique_words)):
    vocab_data[data]=ind

vocab_path.parent.mkdir(parents=True,exist_ok=True)

with open(vocab_path,"w",encoding='utf-8') as file:
    json.dump(vocab_data,file,indent=2,ensure_ascii=False)

print("Vocabulary Size:",len(vocab_data))
print("First Few examples : ",list(vocab_data.items())[:10])