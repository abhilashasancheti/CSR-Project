# This script processes datasets and converts them in the form required for MTL(common, specific)
# and Language Modeling (positive, both)

import os 
import csv
import json
import xml.etree.ElementTree as ET
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# previous mixed up
def convert_JOCI(in_path, mtl_common_path, mtl_specific_path, lm_p_path, lm_both_path):
	with open(in_path) as f, open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_both_path, 'w') as lm2 :
		f = csv.reader(f, delimiter=',')
		i=0
		for row in f:
			if i==0:
				i+=1
				continue
			if row[5].strip() !='COPA':
				premise = row[0].strip().lower()
				hypothesis = row[1].strip().lower()
				label = int(row[2].strip())
				if label>=1 and label<=2:
					label = 0
				else:
					label = 1
				m1.write("{} </s> {} {}\n".format(premise, hypothesis, label))
				m2.write("{} </s> {} {}\n".format(premise, hypothesis, label))
				if label==1:
					lm1.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypothesis))
				else:
					lm2.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypothesis))

def create_context_dict(a):
	i=0
	context_dict = {}
	for row in a:
		if i==0:
			i+=1
			continue
		
		premise = row[0].strip().lower()
		hypothesis = row[1].strip().lower()
		label = int(row[2].strip())
		if premise not in context_dict:
			context_dict[premise] = []

		context_dict[premise].append((hypothesis,label))
	return context_dict

def create_context_dict_j(j, a_dict, b_dict):
	i=0
	context_dict_j = {}
	for row in j:
		if i==0:
			i+=1
			continue
		if row[5].strip() !='COPA':
			premise = row[0].strip().lower()
			hypothesis = row[1].strip().lower()
			label = int(row[2].strip())
			if premise in a_dict or premise in b_dict:
				if premise not in context_dict_j:
					context_dict_j[premise]=[]
				context_dict_j[premise].append((hypothesis,label))

	return context_dict_j

def create_remaining_dict(j, context_dict_j):
	remaining_dict = {}
	i=0
	for row in j:
		if i==0:
			i+=1
			continue
		premise = row[0].strip().lower()
		# print(premise)
		if row[5].strip() !='COPA' and premise not in context_dict_j["train"] and premise not in context_dict_j["test"] and premise not in context_dict_j["dev"]:
			hypothesis = row[1].strip().lower()
			label = int(row[2].strip())
			if premise not in remaining_dict:
				remaining_dict[premise]=[]

			remaining_dict[premise].append((hypothesis,label))

	return remaining_dict

# A and B split train in train, joci-anb is split into train dev and test based on context such that same context appears in one split
def segregate_JOCI_context(in_path):
	# in path is the directory in this case
	with open(in_path + '/' + 'A.train.csv') as a_train, open(in_path + '/' + 'B.train.csv') as b_train, open(in_path + '/' + 'A.test.csv') as a_test, open(in_path + '/' + 'B.test.csv') as b_test, open(in_path + '/' + 'A.dev.csv') as a_dev, open(in_path + '/' + 'B.dev.csv') as b_dev, open(in_path + '/joci-AnB.csv' ) as j:
		a_train = csv.reader(a_train, delimiter=',')
		b_train= csv.reader(b_train, delimiter=',')
		a_test = csv.reader(a_test, delimiter=',')
		b_test= csv.reader(b_test, delimiter=',')
		a_dev = csv.reader(a_dev, delimiter=',')
		b_dev= csv.reader(b_dev, delimiter=',')
		jo = csv.reader(j, delimiter=',')

		context_dict_a = {}
		context_dict_a['train'] = {}
		context_dict_a['test'] = {}
		context_dict_a['dev'] = {}
		context_dict_b = {}
		context_dict_b['train'] = {}
		context_dict_b['test'] = {}
		context_dict_b['dev'] = {}
		context_dict_j = {}
		context_dict_j['train'] = {}
		context_dict_j['test'] = {}
		context_dict_j['dev'] = {}

		remaining_dict = {}

		context_dict_a["train"] = create_context_dict(a_train)
		context_dict_a["test"] = create_context_dict(a_test)
		context_dict_a["dev"] = create_context_dict(a_dev)

		context_dict_b["train"] = create_context_dict(b_train)
		context_dict_b["test"] = create_context_dict(b_test)
		context_dict_b["dev"] = create_context_dict(b_dev)


		context_dict_j["train"] = create_context_dict_j(jo, context_dict_a["train"], context_dict_b["train"])
		context_dict_j["test"] = create_context_dict_j(jo, context_dict_a["test"], context_dict_b["test"])
		context_dict_j["dev"] = create_context_dict_j(jo, context_dict_a["dev"], context_dict_b["dev"])

		jo = csv.reader(open(in_path + '/joci-AnB.csv' ), delimiter=',')
		remaining_dict = create_remaining_dict(jo, context_dict_j)

	with open(in_path+ '/context_a.json', 'w') as f:
		json.dump(context_dict_a,f)

	with open(in_path+ '/context_b.json', 'w') as f:
		json.dump(context_dict_b,f)

	with open(in_path+ '/context_j.json', 'w') as f:
		json.dump(context_dict_j,f)

	with open(in_path + '/remaining_context.json', 'w') as f:
		json.dump(remaining_dict,f)

def write_phase_file(context_dict, m1, m2):
	for premise in context_dict:
		hypotheses = context_dict[premise]
		hypotheses.sort(key = lambda x: x[1], reverse=True) 
		hyps = [h[0] for h in hypotheses]
		labels = [h[1] for h in hypotheses]
		for i in range(len(hyps)):
			lab = labels[i]
			if lab>=1 and lab<=2:
				lab= 0
			else:
				lab= 1
			m1.write("{} </s> {} {}\n".format(premise, hyps[i], lab))
			for j in range(i+1, len(hyps)):
				if labels[i] > labels[j] and labels[i]!=0 and labels[j]!=0:
					lab = np.random.randint(0,2)
					if lab ==0:
						m2.write("{} </s> {} </s> {} {}\n".format(premise, hyps[i], hyps[j], 0))
					else:
						m2.write("{} </s> {} </s> {} {}\n".format(premise, hyps[j], hyps[i], 1))


def convert_JOCI_context(in_path, mtl_common_path, mtl_specific_path, phase='train'):
	with open(in_path + '/remaining_context.json', 'rb') as f, open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(in_path+ '/context_a.json', 'rb') as a, open(in_path+ '/context_b.json', 'rb') as b, open(in_path+ '/context_j.json', 'rb') as j:
		remaining_dict = json.loads(f.read())
		context_dict_a = json.loads(a.read())
		context_dict_b = json.loads(b.read())
		context_dict_j = json.loads(j.read())


		write_phase_file(context_dict_a[phase],m1,m2)
		write_phase_file(context_dict_b[phase],m1,m2)
		write_phase_file(context_dict_j[phase],m1,m2)
		if phase=="train":
			write_phase_file(remaining_dict,m1,m2)

		


def convert_HellaSwag(f, mtl_common_path, mtl_specific_path, lm_p_path, lm_both_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_both_path, 'w') as lm2 :
		for line in f:
			row = json.loads(line.strip())
			premise = row["ctx_a"].encode('utf-8').strip().lower()
			incomplete = row["ctx_b"].encode('utf-8')
			if incomplete:
				incomplete = incomplete.strip().lower()
				hypotheses = [incomplete + ' '+ ending.encode('utf-8').strip().lower() for ending in row["endings"]]
			else:
				hypotheses = [ending.encode('utf-8').strip().lower() for ending in row["endings"]]
			label = row["label"]
			m1.write("{} </s> {} {}\n".format(premise, hypotheses[label], 1))
			neg_hypotheses = [hypotheses[i] for i in range(len(hypotheses)) if i != label]
			m1.write("{} </s> {} </s> {} </s> {} {}\n".format(premise, neg_hypotheses[0], neg_hypotheses[1], neg_hypotheses[2], 0))
			m2.write("{} </s> {} </s> {} </s> {} </s> {} {}\n".format(premise, hypotheses[0], hypotheses[1], hypotheses[2], hypotheses[3], label))
			lm1.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypotheses[label]))
			for i in range(len(hypotheses)):
				if i != label:
					lm2.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypotheses[i]))

def convert_HellaSwag_test(f, mtl_common_path, mtl_specific_path, lm_p_path, lm_both_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_both_path, 'w') as lm2 :
		for line in f:
			row = json.loads(line.strip())
			premise = row["ctx_a"].encode('utf-8').strip().lower()
			incomplete = row["ctx_b"].encode('utf-8')
			if incomplete:
				incomplete = incomplete.strip().lower()
				hypotheses = [incomplete + ' '+ ending.encode('utf-8').strip().lower() for ending in row["endings"]]
			else:
				hypotheses = [ending.encode('utf-8').strip().lower() for ending in row["endings"]]
			label = row["label"]
			m1.write("{} </s> {} {}\n".format(premise, hypotheses[label], 1))
			neg_hypotheses = [hypotheses[i] for i in range(len(hypotheses)) if i != label]
			m1.write("{} </s> {} {}\n".format(premise, neg_hypotheses[0], 0))
			m1.write("{} </s> {} {}\n".format(premise, neg_hypotheses[1], 0))
			m1.write("{} </s> {} {}\n".format(premise, neg_hypotheses[2], 0))
			
			lm1.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypotheses[label]))
			for i in range(len(hypotheses)):
				if i != label:
					lm2.write("[BOS] {} [SEP] {} [EOS]\n".format(premise, hypotheses[i]))
					target = np.random.choice([0,1],1)[0]
					if target==0:
						m2.write("{} </s> {} </s> {} {}\n".format(premise, hypotheses[label], hypotheses[i], target))
					else:
						m2.write("{} </s> {} </s> {} {}\n".format(premise, hypotheses[i], hypotheses[label], target))
	



def convert_aNLI(inp_f, lab_f, mtl_common_path, mtl_specific_path, lm_p_path, lm_n_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_n_path, 'w') as lm2 :
		# inp_f = inp_f.readlines()
		# lab_f = lab_f.readlines()
		for i, line in enumerate(inp_f):
			row = json.loads(line.strip())
			obs1 = row['obs1'].encode('utf-8').strip().lower().replace('\n', ' ')
			obs2 = row['obs2'].encode('utf-8').strip().lower().replace('\n', ' ')
			hyp1 = row['hyp1'].encode('utf-8').strip().lower().replace('\n', ' ')
			hyp2 = row['hyp2'].encode('utf-8').strip().lower().replace('\n', ' ')
			label = lab_f[i].strip()
			if label=="1":
				lm1.write("[BOS] {} {} [SEP] {} [EOS]\n".format(obs1, obs2, hyp1))
				lm2.write("[BOS] {} {} [SEP] {} [EOS]\n".format(obs1, obs2, hyp2))
				m1.write("{} {} </s> {} {}\n".format(obs1, obs2, hyp1, 1))
				m1.write("{} {} </s> {} {}\n".format(obs1, obs2, hyp2, 0))
				m2.write("{} {} </s> {} </s> {} {}\n".format(obs1, obs2, hyp1, hyp2, 0))
			else:
				lm1.write("[BOS] {} {} [SEP] {} [EOS]\n".format(obs1, obs2, hyp2))
				lm2.write("[BOS] {} {} [SEP] {} [EOS]\n".format(obs1, obs2, hyp1))
				m1.write("{} {} </s> {} {}\n".format(obs1, obs2, hyp1, 0))
				m1.write("{} {} </s> {} {}\n".format(obs1, obs2, hyp2, 1))
				m2.write("{} {} </s> {} </s> {} {}\n".format(obs1, obs2, hyp1, hyp2, 1))


def convert_Copa(in_path, mtl_common_path, mtl_specific_path, lm_p_path, lm_n_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_n_path, 'w') as lm2 :
		root = ET.parse(in_path).getroot()
		for child in root:
			label = child.get('most-plausible-alternative')
			obs = child[0].text.encode('utf-8').strip().lower().replace('\n', ' ')
			hyp1 = child[1].text.encode('utf-8').strip().lower().replace('\n', ' ')
			hyp2 = child[2].text.encode('utf-8').strip().lower().replace('\n', ' ')
			if label == "1":
				lm1.write("[BOS] {} [SEP] {} [EOS]\n".format(obs, hyp1))
				lm2.write("[BOS] {} [SEP] {} [EOS]\n".format(obs, hyp2))
				m1.write("{} </s> {} {}\n".format(obs, hyp1, 1))
				m1.write("{} </s> {} {}\n".format(obs, hyp2, 0))
				m2.write("{} </s> {} </s> {} {}\n".format(obs, hyp1, hyp2, 0))
			else:
				lm1.write("[BOS] {} [SEP] {} [EOS]\n".format(obs, hyp2))
				lm2.write("[BOS] {} [SEP] {} [EOS]\n".format(obs, hyp1))
				m1.write("{} </s> {} {}\n".format(obs, hyp1, 0))
				m1.write("{} </s> {} {}\n".format(obs, hyp2, 1))
				m2.write("{} </s> {} </s> {} {}\n".format(obs, hyp1, hyp2, 1))


def convert_Defeasible(list_files, mtl_common_path, mtl_specific_path, lm_p_path, lm_both_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_both_path, 'w') as lm2 :
		for p in list_files:
			with open(p) as f: 
				f = csv.reader(f, delimiter=',')
				i=0
				for row in f:
					if i==0:
						row_head = row
						i+=1
						continue
					l = dict(zip(row_head, row))
					if 'Input_premise' in l.keys():
						premise = l['Input_premise'].strip().lower().replace('\n', ' ')
						hypothesis = l['Input_hypothesis'].strip().lower().replace('\n', ' ')
					else:
						premise = l['Input_situation'].strip().lower().replace('\n', ' ')
						hypothesis = l['Input_rot'].strip().lower().replace('\n', ' ')
					weakener = l['Answer_Attenuator_modifier'].strip().lower().replace('\n', ' ')
					strengthener = l['Answer_Intensifier_modifier'].strip().lower().replace('\n', ' ')
					if weakener:
						lm2.write("[BOS] {} {} [SEP] {} [EOS]\n".format(premise, weakener, hypothesis))
					if strengthener:
						lm1.write("[BOS] {} {} [SEP] {} [EOS]\n".format(premise, strengthener, hypothesis))
					if weakener and strengthener:
						# x = random.random()
						# if x<0.5:
						m1.write("{} {} </s> {} {}\n".format(premise, weakener, hypothesis, 0))
						m1.write("{} {} </s> {} {}\n".format(premise, strengthener, hypothesis, 1))
						m2.write("{} {} </s> {} {}\n".format(premise, weakener, hypothesis, 0))
						m2.write("{} {} </s> {} {}\n".format(premise, strengthener, hypothesis, 1))


def split_hella(in_path, ratio=0.90):
	with open(in_path) as f:
		f = f.readlines()
		random.shuffle(f)
		train_f = f[0: int(ratio*len(f))]
		val_f = f[int(ratio*len(f)):]
	return train_f, val_f

def split_anli(in_path, lab_path, ratio=0.99):
	with open(in_path) as f, open(lab_path) as lab:
		f = f.readlines()
		lab = lab.readlines()
		l = list(zip(f,lab))
		random.shuffle(l)
		f, lab = zip(*l)
		train_f = f[0: int(ratio*len(f))]
		train_lab = lab[0: int(ratio*len(f))]
		val_f = f[int(ratio*len(f)):]
		val_lab = lab[int(ratio*len(f)):]
	return train_f, train_lab, val_f, val_lab

def add_labels(in_path, out_path, label=None):
	with open(in_path, 'r') as inf, open(out_path, 'w') as out:
		lines = inf.readlines()
		for line in lines:
			out.write("{}\n".format(label + ' ' + line.strip()))


def remove_labels(in_path, out_path):
	with open(in_path, 'r') as inf, open(out_path, 'w') as out:
		lines = inf.readlines()
		for line in lines:
			line = line.strip().split(' ')
			line = " ".join(line[1:])
			out.write("{}\n".format(line.strip()))	

#convert_JOCI('./JOCI/data/joci.csv', './JOCI/mtl_common_joci.txt', './JOCI/mtl_specific_joci.txt', './JOCI/lm_p_joci.txt', './JOCI/lm_n_joci.txt')

# train_f, val_f = split_hella('./HellaSwag/hellaswag_train.jsonl')
# convert_HellaSwag( train_f, './HellaSwag/mtl_common_hellaswag_train.txt', './HellaSwag/mtl_specific_hellaswag_train.txt', './HellaSwag/lm_p_hellaswag_train.txt', './HellaSwag/lm_n_hellaswag_train.txt')
# convert_HellaSwag( val_f, './HellaSwag/mtl_common_hellaswag_val.txt', './HellaSwag/mtl_specific_hellaswag_val.txt', './HellaSwag/lm_p_hellaswag_val.txt', './HellaSwag/lm_n_hellaswag_val.txt')
# test_f, val_f = split_hella('./HellaSwag/hellaswag_val.jsonl', ratio=1.0)
# convert_HellaSwag_test( test_f, './HellaSwag/mtl_common_hellaswag_test.txt', './HellaSwag/mtl_specific_hellaswag_test.txt', './HellaSwag/lm_p_hellaswag_test.txt', './HellaSwag/lm_n_hellaswag_test.txt')

# convert_Copa('./COPA/datasets/copa-dev.xml', './COPA/mtl_common_copa_val.txt', './COPA/mtl_specific_copa_val.txt', './COPA/lm_p_copa_val.txt', './COPA/lm_n_copa_val.txt')
# convert_Copa('./COPA/datasets/copa-test.xml', './COPA/mtl_common_copa_test.txt', './COPA/mtl_specific_copa_test.txt', './COPA/lm_p_copa_test.txt', './COPA/lm_n_copa_test.txt')

# train_f, train_lab, val_f, val_lab = split_anli('./aNLI/train.jsonl', './aNLI/train-labels.lst', ratio=0.99)
# print(len(train_f))
# print(len(val_f))
# convert_aNLI(train_f, train_lab, './aNLI/mtl_common_anli_train.txt', './aNLI/mtl_specific_anli_train.txt', './aNLI/lm_p_anli_train.txt', './aNLI/lm_n_anli_train.txt')
# convert_aNLI(val_f, val_lab, './aNLI/mtl_common_anli_val.txt', './aNLI/mtl_specific_anli_val.txt', './aNLI/lm_p_anli_val.txt', './aNLI/lm_n_anli_val.txt')
# test_f, test_lab, empty_f, empty_lab = split_anli('./aNLI/dev.jsonl', './aNLI/dev-labels.lst', ratio=1.0)
# print(len(test_f))
# convert_aNLI(test_f, test_lab, './aNLI/mtl_common_anli_test.txt', './aNLI/mtl_specific_anli_test.txt', './aNLI/lm_p_anli_test.txt', './aNLI/lm_n_anli_test.txt')


# convert_Defeasible(["./defeasible/defeasible-snli/train.csv","./defeasible/defeasible-atomic/train.csv","./defeasible/defeasible-social/train.csv"], './defeasible/mtl_common_defeasible_train.txt', './defeasible/mtl_specific_defeasible_train.txt', './defeasible/lm_p_defeasible_train.txt', './defeasible/lm_n_defeasible_train.txt')
# convert_Defeasible(["./defeasible/defeasible-snli/dev.csv","./defeasible/defeasible-atomic/dev.csv","./defeasible/defeasible-social/dev.csv"], './defeasible/mtl_common_defeasible_val.txt', './defeasible/mtl_specific_defeasible_val.txt', './defeasible/lm_p_defeasible_val.txt', './defeasible/lm_n_defeasible_val.txt')
# convert_Defeasible(["./defeasible/defeasible-snli/test.csv","./defeasible/defeasible-atomic/test.csv","./defeasible/defeasible-social/test.csv"], './defeasible/mtl_common_defeasible_test.txt', './defeasible/mtl_specific_defeasible_test.txt', './defeasible/lm_p_defeasible_test.txt', './defeasible/lm_n_defeasible_test.txt')
# convert_Defeasible(["./defeasible/defeasible-snli/train.csv"], './defeasible/defeasible-snli/mtl_common_snli_train.txt', './defeasible/defeasible-snli/mtl_specific_snli_train.txt', './defeasible/defeasible-snli/lm_p_snlii_train.txt', './defeasible/defeasible-snli/lm_n_snli_train.txt')
# convert_Defeasible(["./defeasible/defeasible-snli/dev.csv"], './defeasible/defeasible-snli/mtl_common_snli_val.txt', './defeasible/defeasible-snli/mtl_specific_snli_val.txt', './defeasible/defeasible-snli/lm_p_snli_val.txt', './defeasible/defeasible-snli/lm_n_snli_val.txt')
# convert_Defeasible(["./defeasible/defeasible-snli/test.csv"], './defeasible/defeasible-snli/mtl_common_snli_test.txt', './defeasible/defeasible-snli/mtl_specific_snli_test.txt', './defeasible/defeasible-snli/lm_p_snli_test.txt', './defeasible/defeasible-snli/lm_n_snli_test.txt')
# convert_Defeasible(["./defeasible/defeasible-atomic/train.csv"], './defeasible/defeasible-atomic/mtl_common_atomic_train.txt', './defeasible/defeasible-atomic/mtl_specific_atomic_train.txt', './defeasible/defeasible-atomic/lm_p_atomic_train.txt', './defeasible/defeasible-atomic/lm_n_atomic_train.txt')
# convert_Defeasible(["./defeasible/defeasible-atomic/dev.csv"], './defeasible/defeasible-atomic/mtl_common_atomic_val.txt', './defeasible/defeasible-atomic/mtl_specific_atomic_val.txt', './defeasible/defeasible-atomic/lm_p_atomic_val.txt', './defeasible/defeasible-atomic/lm_n_atomic_val.txt')
# convert_Defeasible(["./defeasible/defeasible-atomic/test.csv"], './defeasible/defeasible-atomic/mtl_common_atomic_test.txt', './defeasible/defeasible-atomic/mtl_specific_atomic_test.txt', './defeasible/defeasible-atomic/lm_p_atomic_test.txt', './defeasible/defeasible-atomic/lm_n_atomic_test.txt')
# convert_Defeasible(["./defeasible/defeasible-social/train.csv"], './defeasible/defeasible-social/mtl_common_social_train.txt', './defeasible/defeasible-social/mtl_specific_social_train.txt', './defeasible/defeasible-social/lm_p_social_train.txt', './defeasible/defeasible-social/lm_n_social_train.txt')
# convert_Defeasible(["./defeasible/defeasible-social/dev.csv"], './defeasible/defeasible-social/mtl_common_social_val.txt', './defeasible/defeasible-social/mtl_specific_social_val.txt', './defeasible/defeasible-social/lm_p_social_val.txt', './defeasible/defeasible-social/lm_n_social_val.txt')
# convert_Defeasible(["./defeasible/defeasible-social/test.csv"], './defeasible/defeasible-social/mtl_common_social_test.txt', './defeasible/defeasible-social/mtl_specific_social_test.txt', './defeasible/defeasible-social/lm_p_social_test.txt', './defeasible/defeasible-social/lm_n_social_test.txt')


# add_labels("./defeasible/defeasible-snli/mtl_common_snli_train.txt", "./defeasible/defeasible-snli/mtl_common_snli_train_ti.txt", label='[SNLI]')
# add_labels("./defeasible/defeasible-snli/mtl_common_snli_test.txt", "./defeasible/defeasible-snli/mtl_common_snli_test_ti.txt", label='[SNLI]')
# add_labels("./defeasible/defeasible-snli/mtl_common_snli_val.txt", "./defeasible/defeasible-snli/mtl_common_snli_val_ti.txt", label='[SNLI]')

# add_labels("./defeasible/defeasible-atomic/mtl_common_atomic_train.txt", "./defeasible/defeasible-atomic/mtl_common_atomic_train_ti.txt", label='[ATOMIC]')
# add_labels("./defeasible/defeasible-atomic/mtl_common_atomic_test.txt", "./defeasible/defeasible-atomic/mtl_common_atomic_test_ti.txt", label='[ATOMIC]')
# add_labels("./defeasible/defeasible-atomic/mtl_common_atomic_val.txt", "./defeasible/defeasible-atomic/mtl_common_atomic_val_ti.txt", label='[ATOMIC]')

# add_labels("./defeasible/defeasible-social/mtl_common_social_train.txt", "./defeasible/defeasible-social/mtl_common_social_train_ti.txt", label='[SOCIAL]')
# add_labels("./defeasible/defeasible-social/mtl_common_social_test.txt", "./defeasible/defeasible-social/mtl_common_social_test_ti.txt", label='[SOCIAL]')
# add_labels("./defeasible/defeasible-social/mtl_common_social_val.txt", "./defeasible/defeasible-social/mtl_common_social_val_ti.txt", label='[SOCIAL]')

# add_labels("./JOCI/mtl_common_joci_train.txt", "./JOCI/mtl_common_joci_train_ti.txt", label='[JOCI]')
# add_labels("./JOCI/mtl_common_joci_test.txt", "./JOCI/mtl_common_joci_test_ti.txt", label='[JOCI]')
# add_labels("./JOCI/mtl_common_joci_val.txt", "./JOCI/mtl_common_joci_val_ti.txt", label='[JOCI]')

# add_labels("./aNLI/mtl_common_anli_train.txt", "./aNLI/mtl_common_anli_train_ti.txt", label='[ANLI]')
# add_labels("./aNLI/mtl_common_anli_test.txt", "./aNLI/mtl_common_anli_test_ti.txt", label='[ANLI]')
# add_labels("./aNLI/mtl_common_anli_val.txt", "./aNLI/mtl_common_anli_val_ti.txt", label='[ANLI]')

# add_labels("./HellaSwag/mtl_common_hellaswag_train.txt", "./HellaSwag/mtl_common_hellaswag_train_ti.txt", label='[HELLA]')
# add_labels("./HellaSwag/mtl_common_hellaswag_test.txt", "./HellaSwag/mtl_common_hellaswag_test_ti.txt", label='[HELLA]')
# add_labels("./HellaSwag/mtl_common_hellaswag_val.txt", "./HellaSwag/mtl_common_hellaswag_val_ti.txt", label='[HELLA]')

# remove labels to run without labell multi-task model
# remove_labels( "../allCommon/mtl_common_all_train_ti.txt", "../allCommon/mtl_common_all_train.txt")
# remove_labels( "../allCommon/mtl_common_all_val_ti.txt", "../allCommon/mtl_common_all_val.txt")

# create correct joci data
segregate_JOCI_context("./JOCI/data/")
convert_JOCI_context("./JOCI/data/", "./JOCI/mtl_common_joci_train.txt", "./JOCI/mtl_specific_joci_train.txt", phase='train')
convert_JOCI_context("./JOCI/data/", "./JOCI/mtl_common_joci_test.txt", "./JOCI/mtl_specific_joci_test.txt", phase='test')
convert_JOCI_context("./JOCI/data/", "./JOCI/mtl_common_joci_val.txt", "./JOCI/mtl_specific_joci_val.txt", phase='dev')


