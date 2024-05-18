import torch
import os
import argparse
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.optim as optim
import numpy as np

from model import *
from functions import *
from utils import *

def main(args):

    PAD_TOKEN  = torch.nn.CrossEntropyLoss().ignore_index
   
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    preprocessor = Preprocessor(args.tokenizer_type, args.lenght_bert_model)
    
    #Load the data and create dataset and dataloader
    train_raw = load_and_process_data(os.path.join('','ATIS','train.json'))
    test_raw = load_and_process_data(os.path.join('','ATIS','test.json'))

    #Split by creating the Dev set
    Train , Dev , Test  = create_dev(train_raw , test_raw)
    corpus = Train + Dev + Test
    
    #Dataset class -> processing the token
    train = JointDataset(Train , corpus , preprocessor)
    Dev_dataset = JointDataset(Dev , corpus , preprocessor)
    test_dataset = JointDataset(Test , corpus , preprocessor)


    #Mapping of inted and slots
    total_slot_labels = train.get_slots()
    total_intent_labels = train.get_intent()

   
    #Dataloader
    train_dataloader = DataLoader(train, batch_size=args.batch_dimentions_Train)
    dev_dataloader = DataLoader(Dev_dataset, batch_size=args.batch_dimentions_Val_Test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_dimentions_Val_Test)
    
   
    #Model
    model = NerBertModel( args , train.get_slots() , train.get_intent()).to(args.device)


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate  )
  
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    for x in tqdm(range(1, args.N_epochs)):

        loss = train_loop(args , train_dataloader, optimizer, criterion_slots, 
                        criterion_intents, model, total_slot_labels , total_intent_labels)
      
        print("Loss mean" , np.asarray(loss).mean())

        
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            
            print(losses_train)
            
            acc_intet , F1_score = eval_loop(args , dev_dataloader , criterion_slots, criterion_intents, model, total_slot_labels , total_intent_labels , ignore_index=PAD_TOKEN)
            print("intent acc" , np.asarray(acc_intet).mean() ,"F1_score Slots" , np.asarray(F1_score).mean())

    #Vediamo sul Test set   
    acc_intet_test , F1_score_test = eval_loop(args , test_dataloader , criterion_slots, criterion_intents, model, total_slot_labels , total_intent_labels , ignore_index=PAD_TOKEN)
    print("intent acc" , np.asarray(acc_intet_test).mean() , "F1_score Slots" , np.asarray(F1_score_test).mean())

    save_results_to_csv(args ,  np.asarray(F1_score_test).mean() , np.asarray(acc_intet_test).mean() , losses_train , 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--model_type", default="bert-base-uncased", type=str, help="Model type" )
    parser.add_argument("--lenght_bert_model", default="64", type=int, help="Max lenght of the sequence token on Bert" )
    parser.add_argument('--device', type=str, default='cuda',
                        help="Specify the device to use for computation, e.g., 'cuda:0', 'cuda:1', or 'cpu'")
    parser.add_argument("--tokenizer_type", default="bert-base-uncased", type=str, help="Tokenizer type for bert" )

    parser.add_argument("--learning_rate", default="0.01", type=float, help="Learning rate for the model" )


    parser.add_argument("--batch_dimentions_Train", default="64", type=int, help="Dimentions of the Batch_size for the training " )
    parser.add_argument("--batch_dimentions_Val_Test", default="32", type=int, help="Dimentions of the Batch_size for the validation or Test set " )

    parser.add_argument("--N_epochs", default="20", type=int, help="Number of epochs" )
    parser.add_argument("--clip", default="5", type=int, help="Clip of the gradient" )

    parser.add_argument("--Intend_dropout", default="0.1", type=float, help="Dropout LL Intent" )
    parser.add_argument("--Slot_dropout", default="0.1", type=float, help="Dropout LL Slots" )


    args = parser.parse_args()

    
   
    main(args)