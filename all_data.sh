cp aNLI/mtl_specific_anli_train.txt allSpecific/mtl_specific_all_train.txt
cp aNLI/mtl_specific_anli_test.txt allSpecific/mtl_specific_all_test.txt
cp aNLI/mtl_specific_anli_val.txt allSpecific/mtl_specific_all_val.txt
cat defeasible/mtl_specific_defeasible_train.txt >> allSpecific/mtl_specific_all_train.txt
cat defeasible/mtl_specific_defeasible_test.txt >> allSpecific/mtl_specific_all_test.txt
cat defeasible/mtl_specific_defeasible_val.txt >> allSpecific/mtl_specific_all_val.txt
cat HellaSwag/mtl_specific_hellaswag_train.txt >> allSpecific/mtl_specific_all_train.txt
cat HellaSwag/mtl_specific_hellaswag_test.txt >> allSpecific/mtl_specific_all_test.txt
cat HellaSwag/mtl_specific_hellaswag_val.txt >> allSpecific/mtl_specific_all_val.txt
cat JOCI/mtl_specific_joci_train.txt >> allSpecific/mtl_specific_all_train.txt
cat JOCI/mtl_specific_joci_test.txt >> allSpecific/mtl_specific_all_test.txt
cat JOCI/mtl_specific_joci_val.txt >> allSpecific/mtl_specific_all_val.txt

shuf allSpecific/mtl_specific_all_train.txt -o allSpecific/mtl_specific_all_train.txt

# cp aNLI/mtl_common_anli_train.txt allCommon/mtl_common_all_train.txt
# cp aNLI/mtl_common_anli_test.txt allCommon/mtl_common_all_test.txt
# cp aNLI/mtl_common_anli_val.txt allCommon/mtl_common_all_val.txt
# cat defeasible/mtl_common_defeasible_train.txt >> allCommon/mtl_common_all_train.txt
# cat defeasible/mtl_common_defeasible_test.txt >> allCommon/mtl_common_all_test.txt
# cat defeasible/mtl_common_defeasible_val.txt >> allCommon/mtl_common_all_val.txt
# cat HellaSwag/mtl_common_hellaswag_train.txt >> allCommon/mtl_common_all_train.txt
# cat HellaSwag/mtl_common_hellaswag_test.txt >> allCommon/mtl_common_all_test.txt
# cat HellaSwag/mtl_common_hellaswag_val.txt >> allCommon/mtl_common_all_val.txt
# cat JOCI/mtl_common_joci_train.txt >> allCommon/mtl_common_all_train.txt
# cat JOCI/mtl_common_joci_test.txt >> allCommon/mtl_common_all_test.txt
# cat JOCI/mtl_common_joci_val.txt >> allCommon/mtl_common_all_val.txt

# shuf allCommon/mtl_common_all_train.txt -o allCommon/mtl_common_all_train_shuf.txt