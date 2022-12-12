# python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py
import os

domain_list = ['mt_maoyanyanchu', 'mt_taxi-yonghu', 'mt_maicai']
input_folder = "data/sample_datas_wo_prefix/"
output_folder = "output/T5_base_prefix_tuning/single_domain_twice_train/"

for domain_name in domain_list:
    output_path = output_folder + domain_name

    os.system("""
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning_explore
export WANDB_ENTITY=ruotonggeng

CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --run_name %s \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm True \
    --domain_name %s \
    --data_folder_path %s \
    --output_dir %s \
    --seed 2 \
    --cfg Salesforce/T5_base_prefix_summary_3domains_upsample2_prelen25_relu_freeze_plm_mid128.cfg \
    --do_train \
    --do_eval \
    --do_predict \
    --load_weights_from output/T5_base_prefix_tuning/multi_domain_prelen25/checkpoint-21000/pytorch_model.bin \
    --predict_with_generate \
    --num_train_epochs 25 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --metric_for_best_model avr \
    --greater_is_better true \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --adafactor true \
    --learning_rate 1e-3 \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 

""" %(domain_name + 'twice_tune_prelen25', domain_name, input_folder, output_path))