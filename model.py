from seqeval.metrics.sequence_labeling import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import os
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn import metrics
from itertools import chain




class NerBertModel(pl.LightningModule):
    def __init__(self,args ,  slots , intent):
        super().__init__()
        
        self.total_intent_labels = intent
        self.total_slot_labels = slots


        self.bert_config = BertConfig.from_pretrained(args.model_type)
        self.model = BertModel.from_pretrained(args.model_type)

        self.intent_dropout = nn.Dropout(args.Intend_dropout)
        self.intent_linear = nn.Linear(
            self.bert_config.hidden_size, len(self.total_intent_labels)
        )

        self.slot_dropout = nn.Dropout(args.Slot_dropout)
        self.slot_linear = nn.Linear(
            self.bert_config.hidden_size, len(self.total_slot_labels)
        )

       

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        slot_output = outputs[0]
        slot_output = self.slot_dropout(slot_output)
        slot_output = self.slot_linear(slot_output)

        intent_output = outputs[1]
        intent_output = self.intent_dropout(intent_output)
        intent_output = self.intent_linear(intent_output)

        

        return slot_output, intent_output

    
