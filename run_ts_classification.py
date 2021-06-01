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
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from sklearn.metrics import f1_score


from transformers import AutoTokenizer, AutoModel, BertModel,BertTokenizer, RobertaTokenizer, RobertaModel,  BertForSequenceClassification, AutoModelForSequenceClassification, BertConfig
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


logging.basicConfig(level=logging.ERROR)
np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

task_id_2_type = {}
task_id_2_type[0] = "anli"
task_id_2_type[1] = "hella"
task_id_2_type[2] = "copa"
task_id_2_type[3] = "joci"
task_id_2_type[4] = "atomic"
task_id_2_type[5] = "social"
task_id_2_type[6] = "snli"

task_type_2_id = {}
task_type_2_id["anli"] = 0
task_type_2_id["hella"] = 1
task_type_2_id["copa"] = 2
task_type_2_id["joci"] = 3
task_type_2_id["atomic"] = 4
task_type_2_id["social"] = 5
task_type_2_id["snli"] = 6

def create_sub_batches(sequence_output, dataset_type):
    seq_out = {}
    seq_out["anli"] = []
    seq_out["atomic"] = []
    seq_out["copa"] = []
    seq_out["hella"] = []
    seq_out['joci'] = []
    seq_out["snli"] = []
    seq_out["social"] = []
    sub_batches = {}
    sub_batches["anli"] = []
    sub_batches["atomic"] = []
    sub_batches["copa"] = []
    sub_batches["hella"] = []
    sub_batches['joci'] = []
    sub_batches["snli"] = []
    sub_batches["social"] = []

    # print(sequence_output)
    # print(sequence_output[0])
    # segregating instances from same task
    # sequence_output = sequence_output.cpu().detach().numpy()
    for i in range(len(dataset_type)):
        sub_batches[dataset_type[i]].append(int(i))

    for dtype in list(set(dataset_type)):
        seq_out[dtype]= sequence_output[torch.LongTensor(sub_batches[dtype]),:]
    # print(seq_out["anli"])
    return seq_out


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_copa = RobertaClassificationHead(config)
        self.classifier_anli = RobertaClassificationHead(config)
        self.classifier_snli = RobertaClassificationHead(config)
        self.classifier_atomic = RobertaClassificationHead(config)
        self.classifier_social = RobertaClassificationHead(config)
        self.classifier_hella = RobertaClassificationHead(config)
        self.classifier_joci = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        dataset_type=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        seq_out = create_sub_batches(sequence_output, dataset_type)
        logits = None
        
        if len(seq_out['anli'])>0:
            logits = self.classifier_anli(seq_out['anli'])
            # print(logits)
        if len(seq_out['atomic'])>0:
            logits =  torch.cat([logits, self.classifier_atomic(torch.tensor(seq_out['atomic']))],dim=0) if logits is not None else self.classifier_atomic(torch.tensor(seq_out['atomic']))
        if len(seq_out['copa'])>0:
            logits = torch.cat([logits, self.classifier_copa(torch.tensor(seq_out['copa']))],dim=0) if logits is not None else self.classifier_copa(torch.tensor(seq_out['copa']))
        if len(seq_out['hella'])>0:
            logits = torch.cat([logits,self.classifier_hella(torch.tensor(seq_out['hella']))], dim=0) if logits is not None else self.classifier_hella(torch.tensor(seq_out['hella']))
        if len(seq_out['joci'])>0:
            logits = torch.cat([logits,self.classifier_joci(torch.tensor(seq_out['joci']))], dim=0) if logits is not None else self.classifier_joci(torch.tensor(seq_out['joci']))
        if len(seq_out['snli'])>0:
            logits = torch.cat([logits,self.classifier_snli(torch.tensor(seq_out['snli']))], dim=0) if logits is not None else self.classifier_snli(torch.tensor(seq_out['snli']))
        if len(seq_out['social'])>0:
            logits = torch.cat([logits,self.classifier_social(torch.tensor(seq_out['social']))], dim=0) if logits is not None else self.classifier_social(torch.tensor(seq_out['social']))
        
        # print(logits)
        loss = None


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
                
        split_event = event.strip().split(' ')
        event = " ".join(split_event[1:])
        dataset_type = split_event[0].lstrip('[').rstrip(']').lower()
        events = event.strip().split('</s>')
        encoding = self.tokenizer.encode_plus(events[0].strip(), events[1].strip(),add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'type': dataset_type
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



def create_data_loader_test(events, targets, tokenizer, max_len, batch_size):
    
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
    shuffle=False
  )

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

def permute_batch(batch):
    dataset_type = batch['type']
    new_batch = {}
    new_batch['input_ids'] = []
    new_batch['event_description'] = []
    new_batch['attention_mask'] = []
    new_batch['targets'] = []
    new_batch['type'] = []
    permu_mat = torch.zeros((batch["targets"].shape[0],batch["targets"].shape[0]), dtype=batch["targets"].dtype)
    # print("permu mat shape",permu_mat.shape)
    idx = range(len(batch["type"]))
    idx = sorted( idx, key=lambda x : batch["type"][x])
    dataset_type = sorted(dataset_type)
    # print("dataset_type",dataset_type)
    for i,j in enumerate(idx):
        permu_mat[j][i] = 1
    new_batch['type'] = dataset_type
    # print(batch["input_ids"].shape)
    new_batch['input_ids'] = torch.transpose(torch.matmul(torch.transpose(batch["input_ids"],0,1), permu_mat), 0,1)
    # print("input_id_shape", new_batch['input_ids'].shape)
    new_batch["targets"] = torch.matmul(batch["targets"], permu_mat)
    new_batch['attention_mask'] = torch.transpose(torch.matmul(torch.transpose(batch["attention_mask"],0,1), permu_mat),0,1)
    new_batch["event_description"] = [batch['event_description'][i] for i in idx]
    return new_batch


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
    model = model.train()
    i=0
    for d in data_loader:
        d = permute_batch(d)
        #print(d['event_description'], d['targets'])
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        dataset_type = d["type"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, dataset_type=dataset_type)
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
        i+=1
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all), learning_rate

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    f1score = []
    correct_predictions = 0
    
    targets_all, preds_all = [], [] 

    with torch.no_grad():
        for d in data_loader:
            d = permute_batch(d)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            dataset_type = d["type"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, dataset_type=dataset_type)
            outputs = outputs[0]
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive class
            losses.append(loss.item())
            targets_all += targets.cpu().numpy().tolist()
            preds_all += preds.cpu().numpy().tolist()
            
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all)

    
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, dataset_type=dataset_type)
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
    parser.add_argument('-output_dir','--output_dir', type=str, default='./', help="output directory")
    parser.add_argument('--load', action='store_true', help="to load from trained-checkpoint")

    parser.add_argument('-seed', '--seed', type=int, default=42, help="random seed")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=2e-5, help="learning rate")
    parser.add_argument('-decay', '--weight_decay', type=float, default=0.0, help="learning rate")
    parser.add_argument('-warm_up', '--warm_up', type=float, default=0.01, help="lpercentage of warmup steps")
    parser.add_argument('--type', '--type', type=str, default='anli', help="provide type while in test phase")

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
    dataset_type = args.type
    filename = args.filename
    seed = args.seed
    set_seed(seed)
    
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
    
    if do_test:
        model.load_state_dict(torch.load(output_dir+'/best_model_state.bin'))
    elif load_pretrained:
        model.load_state_dict(torch.load(output_dir+'/best_model_state_old.bin'))
        

     
    if do_train:
        train_events, train_targets = read_data(train_data_path)
   
        # load data
        train_data_loader = create_data_loader(train_events, train_targets, tokenizer, max_length, batch_size)
         
    if do_eval:
        val_events, val_targets = read_data(val_data_path)
        
        # load data
        val_data_loader = create_data_loader(val_events, val_targets, tokenizer, max_length, batch_size)
   

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
            optimizer, num_warmup_steps=total_steps*warmup_steps, num_training_steps=total_steps
        )


        # start training
        print("Training starts ....")
        history = defaultdict(list)
        best_score = -1

        model.zero_grad()
        model = model.train()

        # i=0
        # for d in train_data_loader:
        #     print(d)
        #     print(permute_batch(d))
        #     i+=1
        #     if i>1:
        #         break
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
        test_data_loader = create_data_loader_test(test_events, test_targets, tokenizer, max_length, batch_size, dataset_type)

        test_event_texts, test_pred, test_pred_probs, test_test = get_predictions(model, test_data_loader, output_dir, filename)

        
        # with open(output_dir +'/test_events.txt', 'w') as f:
        #     for t in test_event_texts:
        #         f.write("{}\n".format(t.strip()))
        print('--------------')
        # print(test_event_texts[0:5])
        print('-----test report------')
        print(classification_report(test_test,test_pred))
        print(confusion_matrix(test_test, test_pred))
#         print(
