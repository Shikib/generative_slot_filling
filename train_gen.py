import json
import random
import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange

import argparse
import data_readers

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir == '':
        args.output_dir = "models/" + args.train_data_path.replace('/', '_') + '_model_allablation'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

args = read_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").cuda()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

tokenizer.pad_token = tokenizer.eos_token

not_provided = tokenizer.encode("not provided given is")

#  ========================= DATA ==============================
def get_data_utterance(text, slot, answer, req=None):
    if req is not None and len(req) > 0:
	# requested slots
        req_text = "What is the " + ", ".join([slots.get(e) for e in req if e in slots]) + "?"
        #req_text = ", ".join([slots.get(e) for e in req if e in slots]) + " "

        return req_text + " " + tokenizer.eos_token + " " + text + " " + tokenizer.eos_token + " Ok, the {} is {} <|endoftext|>".format(slot, answer)
        #return req_text + " " + tokenizer.eos_token + " " + text + " " + tokenizer.eos_token + " {} is {} <|endoftext|>".format(slot, answer)
    else:
        return text + " " + tokenizer.eos_token + " Ok, the {} is {} <|endoftext|>".format(slot, answer)
        #return text + " " + tokenizer.eos_token + " {} is {} <|endoftext|>".format(slot, answer)

slots = {"time": "time", "people": "number of people", "first_name": "first name", "last_name": "last name", "date": "date", 'pickup_time': 'pickup time', 'to_location': 'going to', 'date': 'date', 'pickup_date': 'pickup date', 'dropoff_date': 'dropoff date', 'subcategory': 'subcategory', 'leaving_date': 'date', 'city_of_event': 'city of the event', 'pickup_city': 'pickup city', 'area': 'area', 'from_location': 'leaving from', 'visit_date': 'visit date'}
#slots = {k:k for k,v in slots.items()} # Only for ablation experiments


def get_data(fn):
    global slots
    examples = data_readers.load_fn(fn)
    seen_slots = set([slot_key for ex in examples for slot_key in ex[2]])
    slots = {k:v for k,v in slots.items() if k in seen_slots}

    data = []
    for ex in tqdm(examples):
        for slot_key,slot_name in slots.items():
            data.append(get_data_utterance(ex[1], slot_name, ex[2].get(slot_key, "not provided"), req=ex[0]))

    return data
    
class GPTDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
    ):
        self.examples = []

        for example in tqdm(data):
            example = " ".join(example.split())
            tokenized = tokenizer(example, padding="max_length", max_length=70)
            self.examples.append(tokenized['input_ids'][-70:])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)



train_data = get_data(args.train_data_path)
train_dataset = GPTDataset(train_data, tokenizer)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=True)
num_epochs = args.num_epochs
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
for epoch in trange(num_epochs, desc="Epoch"):
    model.train()
    epoch_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader):
        # Mask out all irrelevant labels
        labels = batch.clone()
        for i in range(len(labels)):
            # Convert to list
            arr = labels[i].tolist()

            # Find end index
            end_ind = len(arr) - [i for i,e in enumerate(arr[::-1]) if e != 50256][0]

            # Find start_ind
            start_ind = end_ind - arr[:end_ind][::-1].index(318)

            labels[i,:start_ind] = -100
            labels[i,end_ind+2:] = -100

        outputs = model(batch.cuda(), labels=labels.cuda())

        loss, logits = outputs[:2]                        
        (loss/2).backward()

        epoch_loss += loss.item()
        num_batches += 1

        if num_batches % 2 == 1:
            optimizer.step()
            model.zero_grad()

    print("Epoch loss: {}".format(epoch_loss / num_batches))
    if epoch_loss > 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
