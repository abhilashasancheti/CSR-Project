OUTPUT_ANLI_DIR=./model/anli_lm_only_positive
TRAIN_ANLI_FILE=./aNLI_same_order/lm_p_anli_train.txt
VALID_ANLI_FILE=./aNLI_same_order/lm_p_anli_val.txt
# TRAIN_NEG_FILE=./aNLI_same_order/lm_n_anli_train.txt
# TEST_FILE=./aNLI_same_order/lm_n_anli_test.txt
# TEST_DIR=./model/anli_lm_only_positive/test
# MODEL_DIR=./model/anli_lm_only_positive
# TOKENIZER_DIR=./gpt-tokenizer


OUTPUT_JOCI_DIR=./model/joci_lm_only_pos
TRAIN_JOCI_FILE=./JOCI/lm_p_joci_train.txt
VALID_JOCI_FILE=./JOCI/lm_p_joci_val.txt
# TEST_FILE=./JOCI/lm_p_joci_test.txt
# MODEL_DIR=./model/joci_lm_only_pos


OUTPUT_HELLA_DIR=./model/hella_lm_only_pos
TRAIN_HELLA_FILE=./HellaSwag/lm_p_hellaswag_train.txt
VALID_HELLA_FILE=./HellaSwag/lm_p_hellaswag_val.txt

OUTPUT_DEFEASIBLE_ALL_DIR=./model/defeasible_all_only_pos
TRAIN_DEFEASIBLE_ALL_FILE=./defeasible/lm_p_defeasible_train.txt
VALID_DEFEASIBLE_ALL_FILE=./defeasible/lm_p_defeasible_val.txt

OUTPUT_DEFEASIBLE_SOCIAL_DIR=./model/defeasible_social_only_pos
TRAIN_DEFEASIBLE_SOCIAL_FILE=./defeasible/defeasible-social/lm_p_defeasible_train.txt
VALID_DEFEASIBLE_SOCIAL_FILE=./defeasible/defeasible-social/lm_p_defeasible_val.txt

OUTPUT_DEFEASIBLE_SNLI_DIR=./model/defeasible_snli_only_pos
TRAIN_DEFEASIBLE_SNLI_FILE=./defeasible/defeasible-snli/lm_p_defeasible_train.txt
VALID_DEFEASIBLE_SNLI_FILE=./defeasible/defeasible-snli/lm_p_defeasible_val.txt

OUTPUT_DEFEASIBLE_ATOMIC_DIR=./model/defeasible_atomic_only_pos
TRAIN_DEFEASIBLE_ATOMIC_FILE=./defeasible/defeasible-atomic/lm_p_defeasible_train.txt
VALID_DEFEASIBLE_ATOMIC_FILE=./defeasible/defeasible-atomic/lm_p_defeasible_val.txt

# python run_onlypos_langmodel.py --output_dir $OUTPUT_ANLI_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_ANLI_FILE --do_eval --validation_file $VALID_ANLI_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
# python run_onlypos_langmodel.py --output_dir $OUTPUT_JOCI_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_JOCI_FILE --do_eval --validation_file $VALID_JOCI_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
# python run_onlypos_langmodel.py --output_dir $OUTPUT_HELLA_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_HELLA_FILE --do_eval --validation_file $VALID_HELLA_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
# python run_onlypos_langmodel.py --output_dir $OUTPUT_DEFEASIBLE_ALL_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_DEFEASIBLE_ALL_FILE --do_eval --validation_file $VALID_DEFEASIBLE_ALL_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
python run_onlypos_langmodel.py --output_dir $OUTPUT_DEFEASIBLE_SOCIAL_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_DEFEASIBLE_SOCIAL_FILE --do_eval --validation_file $VALID_DEFEASIBLE_SOCIAL_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
python run_onlypos_langmodel.py --output_dir $OUTPUT_DEFEASIBLE_SNLI_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_DEFEASIBLE_SNLI_FILE --do_eval --validation_file $VALID_DEFEASIBLE_SNLI_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16
python run_onlypos_langmodel.py --output_dir $OUTPUT_DEFEASIBLE_ATOMIC_DIR --model_type gpt2 --model_name_or_path gpt2 --do_train --train_file $TRAIN_DEFEASIBLE_ATOMIC_FILE --do_eval --validation_file $VALID_DEFEASIBLE_ATOMIC_FILE --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --evaluate_during_training --learning_rate 5e-5 --num_train_epochs 5 --overwrite_output_dir --gradient_accumulation_steps 16


# python run_onlyneg_langmodel.py --output_dir $OUTPUT_DIR --model_type gpt2 --model_name_or_path gpt2 --tokenizer $TOKENIZER_DIR --do_train --train_pos_file $TRAIN_POS_FILE --train_neg_file $TRAIN_NEG_FILE --do_eval --validation_file $VALID_FILE --batch_size 1  --learning_rate 5e-5 --num_train_epochs 5 --gradient_accumulation_steps 8