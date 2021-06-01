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
import nltk


from nltk.corpus import stopwords 
from collections import defaultdict
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from sklearn.metrics import f1_score


from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, BertForSequenceClassification, RobertaForSequenceClassification, AutoModelForSequenceClassification, BertConfig, XLNetForSequenceClassification, XLNetTokenizer

from transformers.optimization import AdamW, get_linear_schedule_with_warmup



logging.basicConfig(level=logging.ERROR)
np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer),
                 "roberta": (RobertaForSequenceClassification, RobertaTokenizer),
                "xlnet": (XLNetForSequenceClassification, XLNetTokenizer)}

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PlausibleDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len, single):
        self.events = events
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.single = single
        
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
        if self.single:
            encoding = self.tokenizer.encode_plus(event, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
        else:
            events = event.strip().split('</s>')
            encoding = self.tokenizer.encode_plus(events[0].strip(),events[1].strip(), add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def read_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        inputs = []
        labels = []
        for line in lines:
            inputs.append(line.strip()[0:-1].strip())
            labels.append(int(line.strip()[-1]))
    return inputs, labels    

        
def create_data_loader(events, targets, tokenizer, max_len, single, batch_size):
    
    ds = PlausibleDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len,
    single=single
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )

def create_data_loader_test(events, targets, tokenizer, max_len, single, batch_size):
    
    ds = PlausibleDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len,
    single=single
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
  )


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler,
  n_examples
):
    losses = []
    f1score = []
    targets_all, preds_all = [], []
    correct_predictions = 0
    model= model.train()
    for d in data_loader:
        #print(d['event_description'], d['targets'])
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
#         print(input_ids, attention_mask, targets, d['event_description'])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # loss = outputs[0]
        outputs = outputs[0]
        #print(outputs.shape)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive classs
        losses.append(loss.item())
        targets_all += targets.cpu().numpy().tolist()
        preds_all += preds.cpu().numpy().tolist()
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        learning_rate = scheduler.get_last_lr()[0] if version.parse(torch.__version__) >= version.parse("1.4") else scheduler.get_lr()[0]
        
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all, average='weighted'), learning_rate

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    f1score = []
    correct_predictions = 0
    
    targets_all, preds_all = [], [] 

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive class
            losses.append(loss.item())
            targets_all += targets.cpu().numpy().tolist()
            preds_all += preds.cpu().numpy().tolist()
            
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all, average='weighted')

    
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
    print(correct_predictions.double()/len(predictions))
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
    parser.add_argument('-model_path_or_name', '--model_path', required=True, help="path to model")
    parser.add_argument('-tokenizer', '--tokenizer', default="roberta-large", help="path to tokenizer")
    parser.add_argument('-model_type', '--model_type', required=True, help="type of model")
    parser.add_argument('-max_len', '--max_len', type=int, default=128, help="maximum length of the text")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8, help="size of each batch")
    parser.add_argument('-num_classes', '--num_classes', type=int, default=2, help="number of output classes")
    parser.add_argument('-num_train_epochs', '--epochs', type=int, default=20, help="number of trianing epochs")

    parser.add_argument('--do_eval', action='store_true', help="to evaluate")
    parser.add_argument('--do_test', action='store_true', help="to test")
    parser.add_argument('--do_train',action='store_true', help="to train")
    parser.add_argument('--single', action='store_true', help="single input format")
    parser.add_argument('--freeze', action='store_true', help="to train only last layer")
    parser.add_argument('-output_dir','--output_dir', type=str, default='./', help="output directory")
    parser.add_argument('--load', action='store_true', help="to load from trained-checkpoint")
    parser.add_argument('-load_dir','--load_dir', type=str, default='./', help="output directory")

    parser.add_argument('-seed', '--seed', type=int, default=42, help="random seed")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=2e-5, help="learning rate")
    parser.add_argument('-decay', '--weight_decay', type=float, default=0.0, help="learning rate")
    parser.add_argument('-warm_up', '--warm_up', type=float, default=0.01, help="percentage of warmup steps")

    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    max_length = args.max_len
    batch_size = args.batch_size
    num_classes = args.num_classes
    model_type = args.model_type
    epochs = args.epochs
    do_train = args.do_train
    do_test = args.do_test
    output_dir = args.output_dir
    load_pretrained = args.load
    model_path = args.model_path
    learning_rate = args.learning_rate
    dropout = args.dropout
    do_eval = args.do_eval
    adam_epsilon = 1e-8
    warmup_steps = args.warm_up
    filename = args.filename
    load_dir = args.load_dir
    freeze = args.freeze
    seed = args.seed
    single = args.single
    
    set_seed(seed)
    print("working with seed: ", seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    
    if args.tokenizer:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer)
    elif args.model_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer"
        )


    
    model = model_class.from_pretrained(model_path)
    model.config.hidden_dropout_prob = 0.1
    model.attention_probs_dropout_prob = dropout
    model.resize_token_embeddings(len(tokenizer))

    # only train classification layer
    if freeze:
        for param in model.roberta.parameters():
            param.requires_grad = False
    
    if do_test:
        model.load_state_dict(torch.load(output_dir+'/best_model_state.bin'))
    elif load_pretrained:
        model.load_state_dict(torch.load(load_dir+'/best_model_state.bin'))
        

     
    if do_train:
        train_events, train_targets = read_data(train_data_path)
   
        # load data
        train_data_loader = create_data_loader(train_events, train_targets, tokenizer, max_length, single, batch_size)
         
    if do_eval:
        val_events, val_targets = read_data(val_data_path)
        
        # load data
        val_data_loader = create_data_loader(val_events, val_targets, tokenizer, max_length, single,batch_size)
   

    model = model.to(device)
    
    # define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)
 
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
            optimizer, num_warmup_steps=int(total_steps*warmup_steps), num_training_steps=total_steps
        )


        # start training
        print("Training starts ....")
        history = defaultdict(list)
        best_score = -1
        
        model = model.train()
        model.zero_grad()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_acc, train_loss, train_f1score, learning_rate = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_events))

            print(f'Train loss {train_loss} accuracy {train_acc} f1score {train_f1score} lr {learning_rate}')

            val_acc, val_loss, val_f1score = eval_model(model, val_data_loader, loss_fn,  device, len(val_events))
            print(f'Val   loss {val_loss} accuracy {val_acc} f1score {val_f1score}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['train_f1score'].append(train_f1score)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['val_f1score'].append(val_f1score)

            if val_f1score > best_score:
                torch.save(model.state_dict(), output_dir+'/best_model_state.bin')
                best_score = val_f1score
        print("time taken to train", datetime.now()-start)
        
    if do_test: 
        test_events, test_targets = read_data(test_data_path)

        # load data
        test_data_loader = create_data_loader_test(test_events, test_targets, tokenizer, max_length, single, batch_size)

        test_event_texts, test_pred, test_pred_probs, test_test = get_predictions(model, test_data_loader, output_dir, filename)

        
        # with open(output_dir +'/test_events.txt', 'w') as f:
        #     for t in test_event_texts:
        #         f.write("{}\n".format(t.strip()))
        print('--------------')
        #print(test_event_texts[0:5])
        print('-----test report------')
        print(classification_report(test_test,test_pred))
        print(confusion_matrix(test_test, test_pred))
#         print(
