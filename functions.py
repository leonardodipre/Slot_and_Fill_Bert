import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from sklearn.metrics import f1_score as sklearn_f1_score
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_loop(args , data, optimizer, criterion_slots, criterion_intents, model, total_slot_labels , total_intent_labels , metrics ):
    model.train()

    loss_array = []
    loss_slots_array= []
    loss_intent_array = []
    
    for sample in data:

        optimizer.zero_grad() 

        #Extracting all the data usefull for the training
        input_ids, attention_mask, token_type_ids, slot_labels, intent_labels = sample

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        slot_labels = slot_labels.to(args.device)
        intent_labels = intent_labels.to(args.device)

        slots, intent = model(input_ids, attention_mask, token_type_ids)

        
        #For the slots we use the last hidden layer of Bert as output, and using the attention_mask we take in to consideration only the
        #meaninfully part of the output -> using [active_loss]
        active_loss = attention_mask.view(-1) == 1
        active_logits = slots.view(-1, len(total_slot_labels))[active_loss]
        active_labels = slot_labels.view(-1)[active_loss]

        slot_loss = criterion_slots(active_logits, active_labels)
            
        #Intent
        loss_intent = criterion_intents(intent.view(-1, len(total_intent_labels)), intent_labels.view(-1),)

        
        loss =  loss_intent + slot_loss # In joint training we sum the losses. 
                                       # Is there another way to do that?

        loss_slots_array.append(slot_loss.item())
        loss_intent_array.append(loss_intent.item())
        loss_array.append(loss.item())

        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  
        optimizer.step() 

    metrics["loss_slots"]  = np.asarray(loss_slots_array).mean()
    metrics["intent_loss"]  = np.asarray(loss_intent_array).mean()
    metrics["complesive_loss"] = np.asarray(loss_array).mean()




def eval_loop(args , data , criterion_slots, criterion_intents, model, total_slot_labels , total_intent_labels, ignore_index , metrics = None):
        
        model.eval()
        loss_array = []
        acc = []
        F1_score = []
        
        ref_intents = []
        hyp_intents = []
        
        ref_slots = []
        hyp_slots = []
        
        with torch.no_grad(): 

            for sample in data:

                input_ids, attention_mask, token_type_ids, slot_labels, intent_labels = sample

                #PASSO A to Device
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                token_type_ids = token_type_ids.to(args.device)
                slot_labels = slot_labels.to(args.device)
                intent_labels = intent_labels.to(args.device)

                #Modello
                slot_output, intent_output = model(input_ids, attention_mask, token_type_ids)

                
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_output.view(-1, len(total_slot_labels))[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = F.cross_entropy(active_logits, active_labels)

                # extract actual label indices (y_hat) -> pass to cpu from gpu for eval
                a, y_hat = torch.max(slot_output, dim=2)
                y_hat = y_hat.detach().cpu().numpy()
                slot_label_ids = slot_labels.detach().cpu().numpy()



                slot_label_map = {i: label for i, label in enumerate(total_slot_labels)}
                slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
                slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]


                #slot_label_ids -> true indexes 
                #y_hat -> predicetd indextes
                #slot_label_map -> mapping label in to indecies
                for i in range(slot_label_ids.shape[0]): #loop over each sentence
                    for j in range(slot_label_ids.shape[1]): #loopè pver each token

                        # Process only if the token's label is not 'ignore_index'
                        if slot_label_ids[i, j] != ignore_index:
                            # Append the actual label name for the true label index to 'slot_gt_labels'
                            slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                            # Append the actual label name for the predicted label index to 'slot_pred_labels'
                            slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

               

                token_val_slot_acc = sklearn_f1_score(
                    list(chain.from_iterable(slot_gt_labels)),
                    list(chain.from_iterable(slot_pred_labels)),
                    average="micro",
                )

                token_val_slot_acc = torch.tensor(token_val_slot_acc, dtype=torch.float32)
                F1_score.append(token_val_slot_acc.item())

                #INTENT
                intent_loss = criterion_intents( intent_output.view(-1, len(total_intent_labels)), intent_labels.view(-1))
                
                loss_array.append(intent_loss.item())
                
                _, pred_intent_labels = torch.max(intent_output, dim=1)

           
                intent_labels = intent_labels.detach().cpu().numpy()
                pred_intent_labels = pred_intent_labels.detach().cpu().numpy()


                intent_acc = (intent_labels == pred_intent_labels).mean()
                intent_acc = torch.tensor(intent_acc)
                acc.append(intent_acc.item())

        if metrics is not None:
            metrics["Intent_accuracy"] = np.asarray(acc).mean()
            metrics["F1_scores"] = np.asarray(F1_score).mean()
            metrics["Total_loss_eval"] = np.asarray(loss_array).mean() 

        return   acc , F1_score



def plot_metrics(metric_list, filename='metrics_plot.png', training=True):
    # Transforming the dictionary into a DataFrame

    df = pd.DataFrame.from_dict(metric_list, orient='index')

    if training:
        
        # Plotting training metrics
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        df['loss_slots'].plot(ax=axs[0], marker='o', linestyle='-', color='b', title='Loss Slots vs Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss Slots')

        df['intent_loss'].plot(ax=axs[1], marker='o', linestyle='-', color='r', title='Intent Loss vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Intent Loss')

        df['complesive_loss'].plot(ax=axs[2], marker='o', linestyle='-', color='g', title='Complesive Loss vs Epochs')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Complesive Loss')

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    else:
    
        # Plotting evaluation metrics
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))


        df['Intent_accuracy'].plot(ax=axs[0], marker='o', linestyle='-', color='b', title='Intente accuracy vs Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Intent accuracy')

        df['F1_scores'].plot(ax=axs[1], marker='o', linestyle='-', color='r', title='F1  vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('F1 ')

        df['Total_loss_eval'].plot(ax=axs[2], marker='o', linestyle='-', color='g', title='Complesive Loss vs Epochs')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Complesive Loss')

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()