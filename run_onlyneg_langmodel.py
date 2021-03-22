import transformers
import torch
import math
import numpy as np
import random
from torch.nn import functional as F
import logging
import argparse
import os
import sys
from packaging import version
from datetime import datetime
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from collections import defaultdict

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
)

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


class PlausibleDataset(Dataset):
    
    def __init__(self, events, tokenizer, max_len):
        self.events = events
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        event = str(self.events[item])
        # target = self.targets[item]
        

        # split_event = event.strip().split('</s>')
        # if len(split_event)==4:
        #     # input from hellaswag negative datapoint randomly choose one negative answer
        #     hypo = np.random.choice(range(3),1)[0]
        #     event = split_event[0].strip() +  ' </s> ' + split_event[hypo].strip()
        #     target = 0
        encoding = self.tokenizer.encode_plus(event, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': input_ids
    }

def read_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        inputs = []
        # labels = []
        # i=0
        for line in lines:
            inputs.append(line.strip()[0:-1].strip())
            # i +=1
            # if i>2:
            #     break
    return inputs

def create_data_loader(events, tokenizer, max_len, batch_size):
    
    ds = PlausibleDataset(
    events=events,
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )

def train_epoch(
  model, 
  pos_data_loader,  
  neg_data_loader,
  optimizer, 
  device, 
  scheduler,
  n_examples,
  gradient_accumulator_steps
):
    losses = []
    targets_all, preds_all = [], []
    correct_predictions = 0
    neg_data = list(enumerate(neg_data_loader))
    for idx, d in enumerate(pos_data_loader):
        #print(d['event_description'], d['targets'])
        model.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
#         print(input_ids, attention_mask, targets, d['event_description'])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        logits = outputs[1]
        loss = outputs[0]
        # print(loss)
        # loss = outputs[0]
        # outputs = outputs[0]
        #print(outputs.shape)
        # _, preds = torch.max(outputs, dim=1)
        # loss = loss_fn(outputs, targets)
        
        # correct_predictions += torch.sum(preds == targets)
        # f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive classs
        # losses.append(loss.item())
        # targets_all += targets.cpu().numpy().tolist()
        # preds_all += preds.cpu().numpy().tolist()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print(idx, gradient_accumulator_steps, idx+1 % gradient_accumulator_steps)
        if (idx+1) % gradient_accumulator_steps == 0:
            # print("Hi")
            optimizer.step()
            scheduler.step()
        
            model.zero_grad()
            for i in range(gradient_accumulator_steps):
                if idx-i >= len(neg_data):
                    continue
                neg_d = neg_data[idx-i][1]
                input_ids = neg_d["input_ids"].to(device)
                attention_mask = neg_d["attention_mask"].to(device)
                targets = neg_d["targets"].to(device)
        #         print(input_ids, attention_mask, targets, d['event_description'])
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                logits = outputs[1]
                loss = -1*outputs[0]
                # print(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()        
        
        learning_rate = scheduler.get_last_lr()[0] if version.parse(torch.__version__) >= version.parse("1.4") else scheduler.get_lr()[0]
        
    print ("Training done for 1 epoch")

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs[1]
            loss = outputs[0]
            
            losses.append(math.exp(loss/len(input_ids)))
            
    return np.mean(losses)


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Classifier", description="Fine-tunes a given Classifier")
    parser.add_argument('-train_pos_file', '--train_pos_file', required=True, help="path to train positive data file")
    parser.add_argument('-train_neg_file', '--train_neg_file', required=True, help="path to train negative data file")
    parser.add_argument('-test_data_path', '--test_data_path', required=False, help="path to test data file")
    parser.add_argument('-validation_file', '--validation_file', required=True, help="path to val data file")
    parser.add_argument('-model_name_or_path', '--model_name_or_path', required=True, help="path to model")
    parser.add_argument('-tokenizer', '--tokenizer', default="roberta-large", help="path to tokenizer")
    parser.add_argument('-model_type', '--model_type', required=True, help="type of model")
    parser.add_argument('-max_len', '--max_len', type=int, default=128, help="maximum length of the text")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8, help="size of each batch")
    # parser.add_argument('-num_classes', '--num_classes', type=int, default=2, help="number of output classes")
    parser.add_argument('-num_train_epochs', '--num_train_epochs', type=int, default=20, help="number of trianing epochs")

    parser.add_argument('--do_eval', action='store_true', help="to evaluate")
    # parser.add_argument('--do_test', action='store_true', help="to test")
    parser.add_argument('--do_train',action='store_true', help="to train")
    parser.add_argument('-output_dir','--output_dir', type=str, default='./', help="output directory")
    parser.add_argument('--load', action='store_true', help="to load from trained-checkpoint")

    parser.add_argument('-dropout', '--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=2e-5, help="learning rate")
    parser.add_argument('-decay', '--weight_decay', type=float, default=0.0, help="learning rate")
    parser.add_argument('-warm_up', '--warm_up', type=float, default=0.01, help="lpercentage of warmup steps")
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=16, help="steps for backward pass")

    args = parser.parse_args()
    train_pos_data_path = args.train_pos_file
    train_neg_data_path = args.train_neg_file
    # test_data_path = args.test_data_path
    val_data_path = args.validation_file
    max_length = args.max_len
    batch_size = args.batch_size
    # num_classes = args.num_classes
    model_type = args.model_type
    epochs = args.num_train_epochs
    do_train = args.do_train
    # do_test = args.do_test
    output_dir = args.output_dir
    load_pretrained = args.load
    model_path = args.model_name_or_path
    learning_rate = args.learning_rate
    dropout = args.dropout
    do_eval = args.do_eval
    adam_epsilon = 1e-8
    warmup_steps = args.warm_up
    gradient_accumulator_steps = args.gradient_accumulation_steps
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    
    # model_class, tokenizer_class = MODEL_CLASSES[model_type]
    
    # if args.tokenizer:
    #     tokenizer = tokenizer_class.from_pretrained(args.tokenizer)
    # elif args.model_path:
    #     tokenizer = tokenizer_class.from_pretrained(args.model_path)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
    #         "and load it from here, using --tokenizer"
    #     )


    
    # model = model_class.from_pretrained(model_path)
    # model.config.hidden_dropout_prob = 0.1
    # model.attention_probs_dropout_prob = dropout
    # model.resize_token_embeddings(len(tokenizer))
    
    # if do_test:
    #     model.load_state_dict(torch.load(output_dir+'/best_model_state.bin'))
    # elif load_pretrained:
    #     model.load_state_dict(torch.load(output_dir+'/best_model_state_old.bin'))

    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]', 'sep_token': '[SEP]', 'additional_special_tokens': ['[header]','[step]','[title]','[substeps]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
        
    model.resize_token_embeddings(len(tokenizer))
     
    if do_train:
        train_pos_events = read_data(train_pos_data_path)
        train_neg_events = read_data(train_neg_data_path)
   
        # load data
        train_pos_data_loader = create_data_loader(train_pos_events, tokenizer, max_length, batch_size)
        train_neg_data_loader = create_data_loader(train_neg_events, tokenizer, max_length, batch_size)
         
    if do_eval:
        val_events = read_data(val_data_path)
        
        # load data
        val_data_loader = create_data_loader(val_events, tokenizer, max_length, batch_size)
   

    model = model.to(device)
    
 
    if do_train:
        start=datetime.now()
        # define optimizer, and scheduler
        total_steps = len(train_pos_data_loader) * epochs * 2
        
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
        best_score = float("inf")

        model.zero_grad()
        model = model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_perplexity = train_epoch(model, train_pos_data_loader, train_neg_data_loader, optimizer, device, scheduler, len(train_pos_events), gradient_accumulator_steps)

            print(f'Train perplexity {train_perplexity}')

            val_perplexity = eval_model(model, val_data_loader, device, len(val_events))
            print(f'Val   perplexity {val_perplexity}')

            # history['train_acc'].append(train_acc)
            # history['train_loss'].append(train_loss)
            # history['train_f1score'].append(train_f1score)
            # history['val_acc'].append(val_acc)
            # history['val_loss'].append(val_loss)
            # history['val_f1score'].append(val_f1score)

            if val_perplexity < best_score:
                torch.save(model.state_dict(), output_dir+'/best_model_state.bin')
                best_score = val_perplexity
        print("time taken to train", datetime.now()-start)
        
#     if do_test: 
#         test_events, test_targets = read_data(test_data_path)

#         # load data
#         test_data_loader = create_data_loader_test(test_events, test_targets, tokenizer, max_length, batch_size)

#         test_event_texts, test_pred, test_pred_probs, test_test = get_predictions(model, test_data_loader, output_dir, 'test')

        
#         # with open(output_dir +'/test_events.txt', 'w') as f:
#         #     for t in test_event_texts:
#         #         f.write("{}\n".format(t.strip()))
#         print('--------------')
#         print(test_event_texts[0:5])
#         print('-----test report------')
#         print(classification_report(test_test,test_pred))
#         print(confusion_matrix(test_test, test_pred))
# #         print(


# config = AutoConfig.from_pretrained('./model/anli_same_order_lm_only_pos/')
# model = AutoModelForCausalLM.from_pretrained(
#             './model/anli_same_order_lm_only_pos/',
#             from_tf=bool(".ckpt" in './model/anli_same_order_lm_only_pos/'),
#             config=config,
#         )