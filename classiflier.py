import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
import numpy as np
import random

from sklearn.metrics import accuracy_score

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)
 
set_seed(0)
max_len=128

class PlausibleDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len, single=False):
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
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(target, dtype=torch.long)
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

val_data_path = "./JOCI/mtl_common_joci_val.txt"
train_data_path = "./JOCI/mtl_common_joci_train.txt"
val_events, val_targets = read_data(val_data_path)
train_events, train_targets = read_data(train_data_path)


model_name = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained("./roberta-tokenizer/", do_lower_case=True)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")

train_dataset = PlausibleDataset(events=train_events, targets=train_targets, tokenizer=tokenizer, max_len=max_len)
valid_dataset = PlausibleDataset(events=val_events, targets=val_targets, tokenizer=tokenizer, max_len=max_len)


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.0,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=1000,               # log & save weights each logging_steps
    evaluation_strategy="steps",     # evaluate each `logging_steps`
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)


trainer.train()
trainer.evaluate()

model_path="../model/"

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# # reload our model/tokenizer. Optional, only usable when in Python files instead of notebooks
# model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2).to("cuda")
# tokenizer = RobertaTokenizer.from_pretrained(model_path)

# def get_prediction(text):
#     # prepare our text into tokenized sequence
#     inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
#     # perform inference to our model
#     outputs = model(**inputs)
#     # get output probabilities by doing softmax
#     probs = outputs[0].softmax(1)
#     # executing argmax function to get the candidate label
#     return target_names[probs.argmax()]


