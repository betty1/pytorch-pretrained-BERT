#!/bin/bash

for i in $( seq 1 $3)
do
  for t in $( seq 1 20 )
  do
    cd examples
    CUDA_VISIBLE_DEVICES=3 python run_$1.py \
      --bert_model bert-base-uncased \
      --do_lower_case \
      --output_dir "model_dir" \
      --do_predict \
      --predict_file "$2/qa$t/test.json" \
      --eval_single_sample \
      --sample_output_dir "../babi_samples/qa$t_base_$4/qa$t_$i"
    cd ../
    python interpret_layers_babi.py --babi_task "qa$t_base_$4" --sample_name "qa$t_$i"
  done
done
