import dataclasses
import logging
import argparse
import os
import sys
import numpy as np
import random
import json
import torch
from packaging import version
from datetime import datetime
# import nltk


# from nltk.corpus import stopwords 
from collections import defaultdict
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from sklearn.metrics import f1_score


# from transformers import AutoTokenizer, AutoModel, BertModel,BertTokenizer, RobertaTokenizer, RobertaModel,  BertForSequenceClassification, AutoModelForSequenceClassification, BertConfig
# from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead
# from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, BertForSequenceClassification, RobertaForSequenceClassification, AutoModelForSequenceClassification, BertConfig, XLNetForSequenceClassification, XLNetTokenizer

from transformers.optimization import AdamW, get_linear_schedule_with_warmup


np.random.seed(100)
random.seed(100)
torch.manual_seed(100)

logging.basicConfig(level=logging.ERROR)
np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



# class RobertaForSequenceClassification(RobertaPreTrainedModel):
#     authorized_missing_keys = [r"position_ids"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         self.classifier = RobertaClassificationHead(config)
#         self.specific_classifier = RobertaClassificationHead(config)

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         dataset_type=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
#             config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         if dataset_type[0] =='absolute':
#             logits = self.classifier(sequence_output)
#         else:
#             logits = self.specific_classifier(sequence_output)

#         loss = None


#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer),
                 "roberta": (RobertaForSequenceClassification, RobertaTokenizer)}


class PlausibleDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len):
        self.events = events
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        event = str(self.events[item])
        target = self.targets[item]
        

        split_event = event.strip().split('</s>')
        # print(split_event, len(split_event))
        if len(split_event) == 2:
            dataset_type = 'absolute'
        else:
            dataset_type = 'relative'
        # if len(split_event)==5:
        #     # input from hellaswag negative datapoint randomly choose one negative answer
        #     hypo =  np.random.choice([i for i in range(4) if i!=target],1)[0]
        #     label = np.random.choice([0,1], 1)[0]
        #     if label==0:
        #         event = split_event[0].strip() +  ' </s> ' + split_event[target+1].strip() + ' </s> ' + split_event[1+hypo].strip()
        #         target = 0
        #     else:
        #         event = split_event[0].strip() +  ' </s> ' + split_event[1+hypo].strip() + ' </s> ' + split_event[target+1].strip()
        #         target = 1

        if target == 0:
            event_pos = split_event[0].strip() + ' </s> ' + split_event[1].strip()
            event_neg = split_event[0].strip() + ' </s> ' + split_event[2].strip()
        else:
            event_neg = split_event[0].strip() + ' </s> ' + split_event[1].strip()
            event_pos = split_event[0].strip() + ' </s> ' + split_event[2].strip()
        
        encoding_pos = self.tokenizer.encode_plus(event_pos, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids_pos = encoding_pos['input_ids'].flatten()
        encoding_neg = self.tokenizer.encode_plus(event_neg, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids_neg = encoding_neg['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids_pos': input_ids_pos,
      'attention_mask_pos': encoding_pos['attention_mask'].flatten(),
      'input_ids_neg': input_ids_neg,
      'attention_mask_neg': encoding_neg['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'type': dataset_type
    }


class PlausibleDatasetTest(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len):
        self.events = events
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        event = str(self.events[item])
        target = self.targets[item]
        

        split_event = event.strip().split('</s>')
        # for absolute 
        if len(split_event)==4:
            # input from hellaswag negative datapoint randomly choose one negative answer
            hypo = np.random.choice(range(3),1)[0]
            event = split_event[0].strip() +  ' </s> ' + split_event[hypo].strip()
            target = 0
        elif len(split_event)==5:
            # for relative
            # input from hellaswag negative datapoint randomly choose one negative answer
            hypo =  np.random.choice([i for i in range(4) if i!=target],1)[0]
            label = np.random.choice([0,1], 1)[0]
            if label==0:
                event = split_event[0].strip() +  ' </s> ' + split_event[target+1].strip() + ' </s> ' + split_event[1+hypo].strip()
                target = 0
            else:
                event = split_event[0].strip() +  ' </s> ' + split_event[1+hypo].strip() + ' </s> ' + split_event[target+1].strip()
                target = 1
                
        encoding = self.tokenizer.encode_plus(event, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'type': 'absolute'
    }

def read_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        inputs = []
        labels = []
        cnt = 0
        for line in lines:
            inputs.append(line.strip()[0:-1].strip())
            labels.append(int(line.strip()[-1]))
            # if cnt == 10:
            #     break
            # cnt += 1
    return inputs, labels    

        
def create_data_loader(events, targets, tokenizer, max_len, batch_size):
    
    ds = PlausibleDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )

def create_data_loader_test(events, targets, tokenizer, max_len, batch_size):
    
    ds = PlausibleDatasetTest(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
  )


def train_epoch(
  model, 
  data_loader, 
#   loss_fn, 
  optimizer, 
  device, 
  scheduler,
  margin = 0.2
#   n_examples
):
    losses = []
    f1score = []
    targets_all, preds_all = [], []
    # loss_fn = nn.MultiMarginLoss(margin = 0.2).to(device)
    correct_predictions = 0
    cnt = 0
    for d in data_loader:
        #print(d['event_description'], d['targets'])
        # print(cnt)
        # cnt += 1
        optimizer.zero_grad()
        input_ids_pos = d["input_ids_pos"].to(device)
        attention_mask_pos = d["attention_mask_pos"].to(device)
        batch_size = input_ids_pos.shape[0]
        input_ids_neg = d["input_ids_neg"].to(device)
        attention_mask_neg = d["attention_mask_neg"].to(device)
        targets = d["targets"].to(device)
        dataset_type = d["type"]

        # outputs_pos = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos, dataset_type=dataset_type)
        # outputs_neg = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg, dataset_type=dataset_type)
        outputs_pos = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos)
        outputs_neg = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg)
        outputs_pos = outputs_pos[0]
        outputs_neg = outputs_neg[0]
        outputs_diff = outputs_neg-outputs_pos
        outputs_diff = outputs_diff.gather(1, targets.view(-1,1))
    #     _, preds = torch.max(outputs, dim=1)
        # targets  = targets.view(1, targets.shape)
        loss = torch.max(torch.tensor([0]).to(device), torch.tensor([margin]).to(device)+outputs_diff)
        # print(loss)
        loss = torch.sum(loss)
        loss = torch.div(loss, batch_size)
        # print(loss)
        
    #     correct_predictions += torch.sum(preds == targets)
    #     f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive classs
        losses.append(loss.item())
    #     targets_all += targets.cpu().numpy().tolist()
    #     preds_all += preds.cpu().numpy().tolist()
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        learning_rate = scheduler.get_last_lr()[0] if version.parse(torch.__version__) >= version.parse("1.4") else scheduler.get_lr()[0]
        
    # return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all), learning_rate
    return np.mean(losses), learning_rate

def eval_model(model, data_loader, 
    # loss_fn, 
    device, n_examples):
    model = model.eval()
    
    losses = []
    f1score = []
    correct_predictions = 0
    
    targets_all, preds_all = [], [] 

    with torch.no_grad():
        for d in data_loader:
            # input_ids_pos = d["input_ids_pos"].to(device)
            # attention_mask_pos = d["attention_mask_pos"].to(device)
            # input_ids_neg = d["input_ids_neg"].to(device)
            # attention_mask_neg = d["attention_mask_neg"].to(device)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            dataset_type = d["type"]

            # outputs_pos = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos, dataset_type=dataset_type)
            # outputs_neg = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg, dataset_type=dataset_type)
            # outputs_pos = outputs_pos[0]
            # outputs_neg = outputs_neg[0]
            # preds = outputs_pos>outputs_neg
            # preds = preds.gather(1, targets.view(-1,1))
            # loss = torch.max(torch.tensor([0]).to(device), torch.tensor([margin]).to(device)-outputs_pos[0][targets]+outputs_neg[0][targets])
            # loss = torch.sum(loss)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            # print(outputs)
            _, preds = torch.max(outputs, dim=1) #logits in first position of outputs
            # print(preds)
            correct_predictions += torch.sum(preds == targets)

            f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive class
            # losses.append(loss.item())
            targets_all += targets.cpu().numpy().tolist()
            preds_all += preds.cpu().numpy().tolist()
    return correct_predictions.double() / n_examples, f1_score(targets_all, preds_all)       
    # return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all)

    
def get_predictions(model, data_loader, output_dir, split):
    model = model.eval()
    event_descriptions = []
    predictions = []
    prediction_probs = []
    real_values = []
    correct_predictions = 0 
    # with open(output_dir+'/'+split+'_tokens.txt', 'w') as f:
    with torch.no_grad():
        for d in data_loader:
            texts = d["event_description"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            dataset_type = d["type"]
#                 f.write("{}, {}, {}\n".format(texts, input_ids.cpu().detach().numpy(), targets.cpu().detach().numpy()))
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            _, preds = torch.max(outputs, dim=1) #logits in first position of outputs
            correct_predictions += torch.sum(preds == targets)
            probs = F.softmax(outputs, dim=1)
            event_descriptions.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    f1score = f1_score(real_values, predictions, average='weighted') # only for positive classs
    print("accuracy", correct_predictions.double()/len(predictions))
    with open(output_dir+'/'+split+'_predictions.txt', 'w') as f:
        for pred in predictions:
            f.write("{}\n".format(pred))
    with open(output_dir+'/'+ split+ '_prediction_probs.txt', 'w') as f:
        for prob in prediction_probs:
            f.write("{}\n".format(prob.numpy()))
    return event_descriptions, predictions, prediction_probs, real_values


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Classifier", description="Fine-tunes a given Classifier")
    parser.add_argument('-train_data_path', '--train_data_path', required=True, help="path to train data file")
    parser.add_argument('-test_data_path', '--test_data_path', required=True, help="path to test data file")
    parser.add_argument('-val_data_path', '--val_data_path', required=True, help="path to val data file")
    parser.add_argument('-filename', '--filename', default="test", help="name of the prediction file")

    # parser.add_argument('-model_path_or_name', '--model_path', required=True, help="path to model")
    parser.add_argument('-tokenizer', '--tokenizer', default="roberta-large", help="path to tokenizer")
    # parser.add_argument('-model_type', '--model_type', required=True, help="type of model")
    parser.add_argument('-max_len', '--max_len', type=int, default=128, help="maximum length of the text")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8, help="size of each batch")
    parser.add_argument('-num_classes', '--num_classes', type=int, default=2, help="number of output classes")
    parser.add_argument('-num_train_epochs', '--epochs', type=int, default=3, help="number of trianing epochs")

    parser.add_argument('--do_eval', action='store_true', help="to evaluate")
    parser.add_argument('--do_test', action='store_true', help="to test")
    parser.add_argument('--do_train',action='store_true', help="to train")
    parser.add_argument('-output_dir','--output_dir', type=str, default='./', help="output directory")
    parser.add_argument('--load', action='store_true', help="to load from trained-checkpoint")

    parser.add_argument('-dropout', '--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=2e-5, help="learning rate")
    parser.add_argument('-decay', '--weight_decay', type=float, default=0.0, help="learning rate")
    parser.add_argument('-warm_up', '--warm_up', type=float, default=0.01, help="lpercentage of warmup steps")

    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    max_length = args.max_len
    batch_size = args.batch_size
    num_classes = args.num_classes
    epochs = args.epochs
    do_train = args.do_train
    do_test = args.do_test
    output_dir = args.output_dir
    load_pretrained = args.load
    # model_path = args.model_path
    learning_rate = args.learning_rate
    dropout = args.dropout
    do_eval = args.do_eval
    adam_epsilon = 1e-8
    warmup_steps = args.warm_up
    model_type = "roberta"
    filename = args.filename
    # load_dir = args.load_dir

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained('roberta-tokenizer')

    model_path = "roberta-large"
    model = model_class.from_pretrained(model_path)
    model.config.hidden_dropout_prob = 0.1
    model.attention_probs_dropout_prob = dropout
    model.resize_token_embeddings(len(tokenizer))

    if do_test:
        model.load_state_dict(torch.load(output_dir+'/best_model_state.bin'))
    elif load_pretrained:
        model.load_state_dict(torch.load(load_dir+'/best_model_state.bin'))

    if do_train:
        train_events, train_targets = read_data(train_data_path)
        train_data_loader = create_data_loader(train_events, train_targets, tokenizer, max_length, batch_size)

    if do_eval:
        val_events, val_targets = read_data(val_data_path)
        
        # load data
        val_data_loader = create_data_loader_test(val_events, val_targets, tokenizer, max_length, batch_size)


    model = model.to(device)

    if do_train:
        start=datetime.now()
        # define optimizer, and scheduler
        total_steps = len(train_data_loader) * epochs
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps*warmup_steps, num_training_steps=total_steps
        )


        # start training
        print("Training starts ....")
        history = defaultdict(list)
        best_score = -1

        model.zero_grad()
        model = model.train()

        margin = 0.2
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_loss, learning_rate = train_epoch(model, train_data_loader, optimizer, device, scheduler, margin)

            print(f'Train loss {train_loss} lr {learning_rate}')

            # val_acc, val_loss, val_f1score = eval_model(model, val_data_loader,  device, len(val_events))
            val_acc, val_f1score = eval_model(model, val_data_loader,  device, len(val_events))

            print(f'accuracy {val_acc} f1score {val_f1score}')

            # history['train_acc'].append(train_acc)
            # history['train_loss'].append(train_loss)
            # history['train_f1score'].append(train_f1score)
            history['val_acc'].append(val_acc)
            # history['val_loss'].append(val_loss)
            history['val_f1score'].append(val_f1score)

            if val_f1score > best_score:
                torch.save(model.state_dict(), output_dir+'/best_model_state.bin')
                best_score = val_f1score
        print("time taken to train", datetime.now()-start)


    if do_test: 
        test_events, test_targets = read_data(test_data_path)

        # load data
        test_data_loader = create_data_loader_test(test_events, test_targets, tokenizer, max_length, batch_size)

        test_event_texts, test_pred, test_pred_probs, test_test = get_predictions(model, test_data_loader, output_dir, filename)

        
        # with open(output_dir +'/test_events.txt', 'w') as f:
        #     for t in test_event_texts:
        #         f.write("{}\n".format(t.strip()))
        print('--------------')
        #print(test_event_texts[0:5])
        print('-----test report------')
        print(classification_report(test_test,test_pred))
        print(confusion_matrix(test_test, test_pred))

   
