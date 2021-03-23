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
        if dataset_type[0] =='anli':
            logits = self.classifier_anli(sequence_output)
        elif dataset_type[0] =='snli':
            logits = self.classifier_snli(sequence_output)
        elif dataset_type[0] =='atomic':
            logits = self.classifier_atomic(sequence_output)
        elif dataset_type[0] =='social':
            logits = self.classifier_social(sequence_output)
        elif dataset_type[0] =='copa':
            logits = self.classifier_copa(sequence_output)
        elif dataset_type[0] =='joci':
            logits = self.classifier_joci(sequence_output)
        elif dataset_type[0] =='snli':
            logits = self.classifier_snli(sequence_output)
        elif dataset_type[0] =='hella':
            logits = self.classifier_hella(sequence_output)
        else:
            print("invalid type")
            exit()

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

class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size,  bin_size=64, bin_on=False, bin_grow_ratio=0.5):
        self._datasets = datasets
        self._batch_size = batch_size
        # self._mix_opt = mix_opt
        # self._extra_task_ratio = extra_task_ratio
        # self.bin_size = bin_size
        # self.bin_on = bin_on
        # self.bin_grow_ratio = bin_grow_ratio
        train_data_list = []
        for i in range(len(list(datasets.keys()))):
            # if bin_on:
            #     train_data_list.append(self._get_shuffled_index_batches_bin(dataset, batch_size, bin_size=bin_size, bin_grow_ratio=bin_grow_ratio))
            # else:
            train_data_list.append(self._get_shuffled_index_batches(len(datasets[task_id_2_type[i]]), batch_size))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    # @staticmethod
    # def _get_shuffled_index_batches_bin(dataset, batch_size, bin_size, bin_grow_ratio):
    #     maxlen = dataset.maxlen
    #     bins = create_bins(bin_size, maxlen)
    #     data = [[] for i in range(0, len(bins))]
        
    #     for idx, sample in enumerate(dataset):
    #         bin_idx = search_bin(bins, len(sample['sample']['token_id']))
    #         data[bin_idx].append(idx)
    #     index_batches = []

    #     for idx, sub_data in enumerate(data):
    #         if len(sub_data) < 1: continue
    #         batch_size = 1 if batch_size < 1 else batch_size
    #         sub_dataset_len = len(sub_data)
    #         sub_batches = [list(range(i, min(i+batch_size, sub_dataset_len))) for i in range(0, sub_dataset_len, batch_size)]
    #         index_batches.extend(sub_batches)
    #         batch_size = int(batch_size * bin_grow_ratio)
    #     random.shuffle(index_batches)
        # return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list)
        for local_task_idx in all_indices:
            task_id = task_id_2_type[local_task_idx]
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list):
        all_indices = []
        for i in range(0, len(train_data_list)):
            all_indices += [i] * len(train_data_list[i])
    
        random.shuffle(all_indices)
        return all_indices

# class MultiTaskDataset(Dataset):
#     def __init__(self, datasets):
#         self._datasets = datasets
#         task_id_2_data_set_dic = {}
#         for dataset in datasets:
#             task_id = dataset.get_task_id()
#             assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
#             task_id_2_data_set_dic[task_id] = dataset

#         self._task_id_2_data_set_dic = task_id_2_data_set_dic

#     def __len__(self):
#         return sum(len(dataset) for dataset in self._datasets)

#     def __getitem__(self, idx):
#         task_id, sample_id = idx
#         return self._task_id_2_data_set_dic[task_id][sample_id]


class PlausibleDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len, dataset_type):
        self.events = events
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type = dataset_type
        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        event = str(self.events[item])
        target = self.targets[item]
        dataset_type = self.type

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
      'type': dataset_type
    }

class MultiPlausibleDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len):
        self.events = events #dictionary
        self.targets = targets #dictionary
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_types = ['anli', 'hella', 'copa', 'joci', 'atomic', 'social', 'snli' ]
        
    def __len__(self):
        return sum(len(self.events[task_id]) for task_id in self.task_types)

    def __getitem__(self, item):
        task_id, sample_id = item
        event = str(self.events[task_id][sample_id])
        target = self.targets[task_id][sample_id]
        

        split_event = event.strip().split('</s>')
        dataset_type = task_id
                # for absolute 
        if len(split_event)==4:
            # input from hellaswag negative datapoint randomly choose one negative answer
            hypo = np.random.choice(range(3),1)[0]
            event = split_event[0].strip() +  ' </s> ' + split_event[hypo].strip()
            target = 0
        elif len(split_event)==5:
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

def read_data_all(data_path, phase):
    events = {}
    targets = {}
    
    events["anli"], targets["anli"] = read_data(data_path + '/aNLI/mtl_common_anli_'+ phase + '.txt')
    events["hella"], targets["hella"] = read_data(data_path + '/HellaSwag/mtl_common_hellaswag_'+ phase + '.txt')
    events["copa"], targets["copa"] = read_data(data_path + '/COPA/mtl_common_copa_'+ phase + '.txt')
    events["joci"], targets["joci"] = read_data(data_path + '/JOCI/mtl_common_joci_'+ phase + '.txt')
    events["atomic"], targets["atomic"] = read_data(data_path + '/defeasible/defeasible-atomic/mtl_common_atomic_'+ phase + '.txt')
    events["social"], targets["social"] = read_data(data_path + '/defeasible/defeasible-social/mtl_common_social_'+ phase + '.txt')
    events["snli"], targets["snli"] = read_data(data_path + '/defeasible/defeasible-snli/mtl_common_snli_'+ phase + '.txt')

    return events, targets

def create_data_loader_test(events, targets, tokenizer, max_len, batch_size, dataset_type):
    
    ds = PlausibleDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len,
    dataset_type = dataset_type
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False
  )

def create_data_loader(events, targets, tokenizer, max_len, batch_size):
    
    ds = MultiPlausibleDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )
    multi_task_batch_sampler = MultiTaskBatchSampler(events, batch_size)
    
    return DataLoader(ds, batch_sampler=multi_task_batch_sampler)
    



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
    for d in data_loader:
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
        
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all), learning_rate

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
        train_events, train_targets = read_data_all(train_data_path, phase="train")
   
        # load data
        train_data_loader = create_data_loader(train_events, train_targets, tokenizer, max_length, batch_size)
         
    if do_eval:
        val_events, val_targets = read_data_all(val_data_path, phase='val')
        
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
        #     if d['type']==['hella']*batch_size:
        #         print(d)
        #         i+=1
        #     if i>3:
        #         break
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_acc, train_loss, train_f1score, learning_rate = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, sum(len(train_events[e]) for e in train_events))

            print(f'Train loss {train_loss} accuracy {train_acc} f1score {train_f1score} lr {learning_rate}')

            val_acc, val_loss, val_f1score = eval_model(model, val_data_loader, loss_fn,  device, sum(len(val_events[e]) for e in val_events))
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
