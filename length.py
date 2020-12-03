import os
import numpy as np
def calc_length(path):
	with open(path) as f:
		lines = f.readlines()
		lines = [line.strip() for line in lines]
	length = [len(line.split()) for line in lines]
	return np.min(length), np.max(length), np.mean(length)


print("hella_p", calc_length('./HellaSwag/mtl_common_hellaswag_train.txt'))
print("joci", calc_length('./JOCI/mtl_common_joci_train.txt'))
print("anli", calc_length('./aNLI/mtl_common_anli_train.txt'))
print("defeasible", calc_length('./defeasible/mtl_common_defeasible_train.txt'))