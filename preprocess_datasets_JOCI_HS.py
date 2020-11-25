# This script processes datasets and converts them in the from requirred for MTL(common, specific)
# and Language Modeling (positive, both)

import os 
import csv
import json
import xml.etree.ElementTree as ET
import random
random.seed(42)

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


def convert_aNLI(inp_f, lab_f, mtl_common_path, mtl_specific_path, lm_p_path, lm_n_path):
	with open(mtl_common_path, 'w') as m1, open(mtl_specific_path, 'w') as m2, open(lm_p_path, 'w') as lm1, open(lm_n_path, 'w') as lm2 :
		# inp_f = inp_f.readlines()
		# lab_f = lab_f.readlines()
		for i, line in enumerate(inp_f):
			row = json.loads(line.strip())
			obs1 = row['obs1'].encode('utf-8').strip().lower()
			obs2 = row['obs2'].encode('utf-8').strip().lower()
			hyp1 = row['hyp1'].encode('utf-8').strip().lower()
			hyp2 = row['hyp2'].encode('utf-8').strip().lower()
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
			obs = child[0].text.encode('utf-8').strip().lower()
			hyp1 = child[1].text.encode('utf-8').strip().lower()
			hyp2 = child[2].text.encode('utf-8').strip().lower()
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
						premise = l['Input_premise']
						hypothesis = l['Input_hypothesis']
					else:
						premise = l['Input_situation']
						hypothesis = l['Input_rot']
					weakener = l['Answer_Attenuator_modifier']
					strengthener = l['Answer_Intensifier_modifier']
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

#convert_JOCI('./JOCI/data/joci.csv', './JOCI/mtl_common_joci.txt', './JOCI/mtl_specific_joci.txt', './JOCI/lm_p_joci.txt', './JOCI/lm_n_joci.txt')

# train_f, val_f = split_hella('./HellaSwag/hellaswag_train.jsonl')
# convert_HellaSwag( train_f, './HellaSwag/mtl_common_hellaswag_train.txt', './HellaSwag/mtl_specific_hellaswag_train.txt', './HellaSwag/lm_p_hellaswag_train.txt', './HellaSwag/lm_n_hellaswag_train.txt')
# convert_HellaSwag( val_f, './HellaSwag/mtl_common_hellaswag_val.txt', './HellaSwag/mtl_specific_hellaswag_val.txt', './HellaSwag/lm_p_hellaswag_val.txt', './HellaSwag/lm_n_hellaswag_val.txt')
# test_f, val_f = split_hella('./HellaSwag/hellaswag_val.jsonl', ratio=1.0)
# convert_HellaSwag( test_f, './HellaSwag/mtl_common_hellaswag_test.txt', './HellaSwag/mtl_specific_hellaswag_test.txt', './HellaSwag/lm_p_hellaswag_test.txt', './HellaSwag/lm_n_hellaswag_test.txt')

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
convert_Defeasible(["./defeasible/defeasible-snli/train.csv"], './defeasible/defeasible-snli/mtl_common_defeasible_train.txt', './defeasible/defeasible-snli/mtl_specific_defeasible_train.txt', './defeasible/defeasible-snli/lm_p_defeasible_train.txt', './defeasible/defeasible-snli/lm_n_defeasible_train.txt')
convert_Defeasible(["./defeasible/defeasible-snli/dev.csv"], './defeasible/defeasible-snli/mtl_common_defeasible_val.txt', './defeasible/defeasible-snli/mtl_specific_defeasible_val.txt', './defeasible/defeasible-snli/lm_p_defeasible_val.txt', './defeasible/defeasible-snli/lm_n_defeasible_val.txt')
convert_Defeasible(["./defeasible/defeasible-snli/test.csv"], './defeasible/defeasible-snli/mtl_common_defeasible_test.txt', './defeasible/defeasible-snli/mtl_specific_defeasible_test.txt', './defeasible/defeasible-snli/lm_p_defeasible_test.txt', './defeasible/defeasible-snli/lm_n_defeasible_test.txt')
convert_Defeasible(["./defeasible/defeasible-atomic/train.csv"], './defeasible/defeasible-atomic/mtl_common_defeasible_train.txt', './defeasible/defeasible-atomic/mtl_specific_defeasible_train.txt', './defeasible/defeasible-atomic/lm_p_defeasible_train.txt', './defeasible/defeasible-atomic/lm_n_defeasible_train.txt')
convert_Defeasible(["./defeasible/defeasible-atomic/dev.csv"], './defeasible/defeasible-atomic/mtl_common_defeasible_val.txt', './defeasible/defeasible-atomic/mtl_specific_defeasible_val.txt', './defeasible/defeasible-atomic/lm_p_defeasible_val.txt', './defeasible/defeasible-atomic/lm_n_defeasible_val.txt')
convert_Defeasible(["./defeasible/defeasible-atomic/test.csv"], './defeasible/defeasible-atomic/mtl_common_defeasible_test.txt', './defeasible/defeasible-atomic/mtl_specific_defeasible_test.txt', './defeasible/defeasible-atomic/lm_p_defeasible_test.txt', './defeasible/defeasible-atomic/lm_n_defeasible_test.txt')
convert_Defeasible(["./defeasible/defeasible-social/train.csv"], './defeasible/defeasible-social/mtl_common_defeasible_train.txt', './defeasible/defeasible-social/mtl_specific_defeasible_train.txt', './defeasible/defeasible-social/lm_p_defeasible_train.txt', './defeasible/defeasible-social/lm_n_defeasible_train.txt')
convert_Defeasible(["./defeasible/defeasible-social/dev.csv"], './defeasible/defeasible-social/mtl_common_defeasible_val.txt', './defeasible/defeasible-social/mtl_specific_defeasible_val.txt', './defeasible/defeasible-social/lm_p_defeasible_val.txt', './defeasible/defeasible-social/lm_n_defeasible_val.txt')
convert_Defeasible(["./defeasible/defeasible-social/test.csv"], './defeasible/defeasible-social/mtl_common_defeasible_test.txt', './defeasible/defeasible-social/mtl_specific_defeasible_test.txt', './defeasible/defeasible-social/lm_p_defeasible_test.txt', './defeasible/defeasible-social/lm_n_defeasible_test.txt')






