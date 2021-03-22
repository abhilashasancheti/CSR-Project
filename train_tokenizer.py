from transformers import RobertaTokenizer, GPT2Tokenizer
import os


if not os.path.exists('./roberta-tokenizer/'):
	os.mkdir('./roberta-tokenizer/')

if not os.path.exists('./gpt-tokenizer/'):
	os.mkdir('./gpt-tokenizer/')

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
special_tokens_dict = {'additional_special_tokens': ['[title]','[step]','[header]', '[substeps]', '[ANLI]', '[HELLA]','[JOCI]', '[COPA]', '[SOCIAL]', '[SNLI]', '[ATOMIC]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained('./roberta-tokenizer/')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]', 'sep_token': '[SEP]', 'additional_special_tokens': ['[header]','[step]','[title]','[substeps]', '[ANLI]', '[HELLA]','[JOCI]', '[COPA]', '[SOCIAL]', '[SNLI]', '[ATOMIC]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained('./gpt-tokenizer/')
