#cp aNLI/mtl_specific_anli_train.txt allSpecific/mtl_specific_all_train.txt
#cp aNLI/mtl_specific_anli_test.txt allSpecific/mtl_specific_all_test.txt
#cp aNLI/mtl_specific_anli_val.txt allSpecific/mtl_specific_all_val.txt
#cat defeasible/mtl_specific_defeasible_train.txt >> allSpecific/mtl_specific_all_train.txt
#cat defeasible/mtl_specific_defeasible_test.txt >> allSpecific/mtl_specific_all_test.txt
#cat defeasible/mtl_specific_defeasible_val.txt >> allSpecific/mtl_specific_all_val.txt
#cat HellaSwag/mtl_specific_hellaswag_train.txt >> allSpecific/mtl_specific_all_train.txt
#cat HellaSwag/mtl_specific_hellaswag_test.txt >> allSpecific/mtl_specific_all_test.txt
#cat HellaSwag/mtl_specific_hellaswag_val.txt >> allSpecific/mtl_specific_all_val.txt
#cat JOCI/mtl_specific_joci_train.txt >> allSpecific/mtl_specific_all_train.txt
#cat JOCI/mtl_specific_joci_test.txt >> allSpecific/mtl_specific_all_test.txt
#cat JOCI/mtl_specific_joci_val.txt >> allSpecific/mtl_specific_all_val.txt

#shuf allSpecific/mtl_specific_all_train.txt -o allSpecific/mtl_specific_all_train.txt

# cp aNLI/mtl_common_anli_train_ti.txt ../allCommon/mtl_common_all_train_ti.txt
# cp aNLI/mtl_common_anli_test_ti.txt ../allCommon/mtl_common_all_test_ti.txt
# cp aNLI/mtl_common_anli_val_ti.txt ../allCommon/mtl_common_all_val_ti.txt
# cat defeasible/defeasible-snli/mtl_common_snli_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat defeasible/defeasible-snli/mtl_common_snli_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat defeasible/defeasible-snli/mtl_common_snli_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt
# cat defeasible/defeasible-social/mtl_common_social_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat defeasible/defeasible-social/mtl_common_social_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat defeasible/defeasible-social/mtl_common_social_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt
# cat defeasible/defeasible-atomic/mtl_common_atomic_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat defeasible/defeasible-atomic/mtl_common_atomic_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat defeasible/defeasible-atomic/mtl_common_atomic_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt
# cat HellaSwag/mtl_common_hellaswag_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat HellaSwag/mtl_common_hellaswag_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat HellaSwag/mtl_common_hellaswag_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt
# cat JOCI/mtl_common_joci_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat JOCI/mtl_common_joci_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat JOCI/mtl_common_joci_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt
# cat COPA/mtl_common_copa_train_ti.txt >> ../allCommon/mtl_common_all_train_ti.txt
# cat COPA/mtl_common_copa_test_ti.txt >> ../allCommon/mtl_common_all_test_ti.txt
# cat COPA/mtl_common_copa_val_ti.txt >> ../allCommon/mtl_common_all_val_ti.txt

# shuf ../allCommon/mtl_common_all_train_ti.txt -o ../allCommon/mtl_common_all_train_ti_shuf.txt
python run_ts_classification.py -train_data_path ./ -val_data_path ./  -test_data_path ./JOCI/mtl_common_all_test.txt  -model_path_or_name=roberta-large -model_type roberta --do_train --do_eval -output_dir ../roberta-common-all-ts-e3 -tokenizer ./roberta-tokenizer/ -num_train_epochs 3 -batch_size 16 > ./outputs/all_common_ts_e3.txt


