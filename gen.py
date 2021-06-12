from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

import json
import os
import torch
import numpy as np
import data_readers

import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    args = parser.parse_args()

    if args.output_dir == '':
        args.output_dir = "models/" + args.train_data_path.replace('/', '_') + '_model_allablation'
    
    return args

args = read_args()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").cuda()
#tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
#model = AutoModelForCausalLM.from_pretrained("gpt2-medium").cuda()
model.load_state_dict(torch.load("{}/model.pt".format(args.output_dir)))
model.eval()

not_provided = tokenizer.encode("not provided given is")

all_vocab = [tokenizer.decode([e]) for e in range(tokenizer.vocab_size)]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def closest(pred, text):
    # try all subsets of the text
    words = text.split()
    best, best_score = "", 100
    for start in range(len(words)):
        for end in range(start+1, len(words)+1):
            subset = " ".join(words[start:end])
            score = levenshteinDistance(pred, subset)
            if score < best_score:
                best, best_score = subset, score

    if best_score < 0.3*len(pred):
        return best
    else:
        return ""

def predict(text, slot, req=None, debug=False):
    if req is not None and len(req) > 0:
	# requested slots
        req_text = "What is the " + ", ".join([slots.get(e) for e in req if e in slots]) + "?"
        #req_text =  ", ".join([slots.get(e) for e in req if e in slots]) + " "

        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(req_text + " " + tokenizer.eos_token + " " + text + " " + tokenizer.eos_token + " Ok, the {} is".format(slot), return_tensors='pt')
        #new_user_input_ids = tokenizer.encode(req_text + " " + tokenizer.eos_token + " " + text + " " + tokenizer.eos_token + " {} is".format(slot), return_tensors='pt')
    else:
        new_user_input_ids = tokenizer.encode(text + " " + tokenizer.eos_token + " Ok, the {} is".format(slot), return_tensors='pt')
        #new_user_input_ids = tokenizer.encode(text + " " + tokenizer.eos_token + " {} is".format(slot), return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids

    #only_text_input_ids = tokenizer.encode(text + " " + tokenizer.eos_token)
    good_words = set([i for i,e in enumerate(all_vocab) if e.strip() in text + " <|endoftext|> is not provided not given"])

    ## generated a response while limiting the total chat history to 1000 tokens, 
    bad_words = [[e] for e in range(tokenizer.vocab_size) if e not in good_words]

    #bad_words = [[e] for e in range(tokenizer.vocab_size) if e not in list(only_text_input_ids + not_provided)]
    #import pdb; pdb.set_trace()

    #chat_history_ids = model.generate(bot_input_ids.cuda(), max_length=2*bot_input_ids.size(1), pad_token_id=tokenizer.eos_token_id, bad_words=bad_words)
    chat_history_ids = model.generate(bot_input_ids.cuda(), max_length=2*bot_input_ids.size(1), pad_token_id=tokenizer.eos_token_id)#, bad_words_ids=bad_words)


    # pretty print last ouput tokens from bot
    #print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, ][0])))
    pred = tokenizer.decode(chat_history_ids[:, bot_input_ids.size(1): ][0], skip_special_tokens=True)
    if debug:
      print(pred, "||", text, "||", tokenizer.decode(chat_history_ids[0]))
    if pred.strip() not in text:
      # see if we can find edit distance?
      return closest(pred.strip(), text)
    else:
      return pred.strip()

def evaluate(pred, true, slot_types=["time", "people", "first_name", "last_name", "date"]):
    slot_type_f1_scores = []

    for slot_type in slot_types:
        predictions_for_slot = [
            [p[1] for p in prediction.items() if p[0] == slot_type] for prediction in pred
        ]
        labels_for_slot = [
            [l[1] for l in label.items() if l[0] == slot_type] for label in true
        ]

        proposal_made = [len(p) > 0 for p in predictions_for_slot]
        has_label = [len(l) > 0 for l in labels_for_slot]
        prediction_correct = [
            prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
        ]
        true_positives = sum([
            int(proposed and correct)
            for proposed, correct in zip(proposal_made, prediction_correct)
        ])
        num_predicted = sum([int(proposed) for proposed in proposal_made])
        num_to_recall = sum([int(hl) for hl in has_label])

        precision = true_positives / (1e-5 + num_predicted)
        recall = true_positives / (1e-5 + num_to_recall)

        f1_score = 2 * precision * recall / (1e-5 + precision + recall)
        slot_type_f1_scores.append(f1_score)

        print(slot_type, precision, recall, f1_score)

    return np.mean(slot_type_f1_scores)

examples = data_readers.load_fn(args.test_data_path)

seen_slots = set([slot_key for ex in examples for slot_key in ex[2]])
results = []
slots = {"time": "time", "people": "number of people", "first_name": "first name", "last_name": "last name", "date": "date", 'pickup_time': 'pickup time', 'to_location': 'going to', 'date': 'date', 'pickup_date': 'pickup date', 'dropoff_date': 'dropoff date', 'subcategory': 'subcategory', 'leaving_date': 'date', 'city_of_event': 'city of the event', 'pickup_city': 'pickup city', 'area': 'area', 'from_location': 'leaving from', 'visit_date': 'visit date'}

#slots = {k:k for k,v in slots.items()} # For ablation experiments

slots = {k:v for k,v in slots.items() if k in seen_slots}
all_preds = []
all_true = []
for ex in tqdm(examples):
    pred_slots = {}
    for slot_key,slot_name in slots.items():
        pred = predict(ex[1], slot_name, req=ex[0])
        if pred != "":
            pred_slots[slot_key] = pred

    all_preds.append(pred_slots)
    all_true.append(ex[2])

    if len(all_preds) % 100 == 0:
        score = evaluate(all_preds, all_true, slot_types=seen_slots)

score = evaluate(all_preds, all_true, slot_types=seen_slots)
print("FINAL SCORE:", score)
open(args.output_dir + "/scores.txt", "w+").write(str(score))
