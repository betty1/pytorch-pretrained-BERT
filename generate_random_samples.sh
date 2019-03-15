#!/bin/bash

for i in $( seq 1 $3)
do
  for t in $( seq 1 20 )
  do
    cd examples
    CUDA_VISIBLE_DEVICES=0 python run_$1.py \
      --bert_model bert-base-uncased \
      --do_lower_case \
      --output_dir "model_dir" \
      --do_predict \
      --predict_file "${2}/qa${t}/test.json" \
      --eval_single_sample \
      --sample_output_dir "../babi_samples/base_${4}/qa${t}_${i}"
    cd ../
    if [ $1 = "babi" ]
    then
    	python interpret_layers_babi.py --babi_task "base_${4}" --sample_name "qa${t}_${i}" --sentence_colored
	else
		python interpret_layers_hotpot.py --babi_task "base_${4}" --sample_name "qa${t}_${i}" --sentence_colored
	fi
  done
done
