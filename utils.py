import torch
import json
import csv
import os
from argparse import Namespace
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch.utils.data as data



def load_and_process_data(path):
    # Load the JSON data from the file
    with open(path, 'r') as f:
        dataset = json.load(f)
    
    # Process each entry in the dataset
    processed_data = []
    for entry in dataset:
        # Extract utterance, intent, and slots
        utterance = entry['utterance']
        intent = entry['intent']
        # Split slots on whitespace, treating them as separate tokens
        slots = entry['slots'].split()

        # Append the processed data to the list
        processed_data.append({'utterance': utterance, 'intent': intent, 'slots': slots})
    
    return processed_data


def create_dev(tmp_train_raw, test_raw):

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train

    dev_raw = X_dev

    #Get only intent from test
    #y_test = [x['intent'] for x in test_raw]

    y_test = test_raw

    return train_raw , dev_raw , y_test 


class JointDataset(data.Dataset):
    def __init__(self, dataset, corpus , preprocessor):
        
        self.preprocessor = preprocessor

        self.utterances = []
        self.slots = []
        self.intents = []
        

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])



        self.slots_tok = list(set(sum([line['slots'] for line in corpus],[])))
        self.slot_labels = sorted(self.slots_tok, key=lambda x: (x[2:], x[:2]))
        self.slot_labels = ["UNK", "PAD"] + self.slot_labels


        self.intent_tok  = list(set([line['intent'] for line in corpus]))
        self.intent_labels = sorted(self.intent_tok)
        self.intent_labels = ["UNK"] + self.intent_labels

    def __len__(self):
        return len(self.utterances)

    def get_slots(self):
        return self.slot_labels

    def get_intent(self):
        return self.intent_labels

    def __getitem__(self, idx):

        utterance = self.utterances[idx].split()

        intent = self.intents[idx]

        intent_id = (
            self.intent_labels.index(intent)
            if intent in self.intent_labels
            else self.intent_labels.index("UNK")
        )


        slot = self.slots[idx]

        slot_id = [
            self.slot_labels.index(t)
            if t in self.slot_labels
            else self.slot_labels.index("UNK")

            for t in self.slots[idx]
        ]

        
        return self.preprocessor.get_input_features(utterance, slot_id, intent_id) 




class Preprocessor:
    def __init__(self, model_type, max_len):
        
        self.tokenizer = BertTokenizer.from_pretrained(model_type)
        self.max_len = max_len
        self.ignore_index = torch.nn.CrossEntropyLoss().ignore_index

    def get_input_features(self, sentence, tags, intent):



        input_tokens = []
        slot_labels = []

        for word, tag in zip(sentence, tags):
            tokens = self.tokenizer.tokenize(word)

            if len(tokens) == 0:
                tokens = self.tokenizer.unk_token

            input_tokens.extend(tokens)

            for i, _ in enumerate(tokens):
                if i == 0:
                    slot_labels.extend([tag])
                else:
                    slot_labels.extend([self.ignore_index])

        # 2. max_len
        if len(input_tokens) > self.max_len - 2:
            input_tokens = input_tokens[: self.max_len - 2]
            slot_labels = slot_labels[: self.max_len - 2]

      
        input_tokens = (
            [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        )
        slot_labels = [self.ignore_index] + slot_labels + [self.ignore_index]

        # token
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # padding
        pad_len = self.max_len - len(input_tokens)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * pad_len)
        slot_labels = slot_labels + ([self.ignore_index] * pad_len)
        attention_mask = attention_mask + ([0] * pad_len)
        token_type_ids = token_type_ids + ([0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        slot_labels = torch.tensor(slot_labels, dtype=torch.long)

        
        intent = torch.tensor(intent, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, slot_labels, intent


def save_results_to_csv(args, f1_score, accuracy_intent, loss_test, losses_train, filepath='model_results.csv'):
    # Convert Namespace args to dictionary
    args_dict = vars(args)
    
    # Add the test scores and average training loss to the dictionary
    args_dict['f1_score'] = f1_score
    args_dict['accuracy_intent'] = accuracy_intent
    args_dict['loss_test'] = loss_test
    args_dict['average_loss_train'] = sum(losses_train) / len(losses_train) if losses_train else 0  # Compute average if list is not empty

    # Check if the file exists
    file_exists = os.path.isfile(filepath)
    
    # Open the file in append mode
    with open(filepath, mode='a', newline='') as file:
        # Create a writer object from csv module
        writer = csv.DictWriter(file, fieldnames=args_dict.keys())
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(args_dict)