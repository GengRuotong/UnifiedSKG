# python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py
import os

input_path = ["data/sample_datas_wo_prefix/", "data/sample_datas_w_prefix_ahead/"]
output_path = ["output/T5_base_ft_wo_prefix/multi_domain_mix/mt_multi", "output/T5_base_ft_w_prefix_ahead/multi_domain_mix/mt_multi"]

for i in range(2):
    os.system("""
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_ft_wo_prefix
export WANDB_ENTITY=ruotonggeng

python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \
    --run_name mt_multi_wo_prefix_mix \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm False \
    --domain_name mt_multi \
    --data_folder_path %s \
    --output_dir %s \
    --seed 2 \
    --cfg Salesforce/T5_base_finetune_summary_all_domains_upsample1.cfg \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --num_train_epochs 5 \
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
    --learning_rate 1e-4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1
""" %(input_path[0], output_path[0]))
        
    os.system("""
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_ft_w_prefix
export WANDB_ENTITY=ruotonggeng

python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \
    --run_name mt_multi_w_prefix_mix \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm False \
    --domain_name mt_multi \
    --data_folder_path %s \
    --output_dir %s \
    --seed 2 \
    --cfg Salesforce/T5_base_finetune_summary_all_domains_upsample1.cfg \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
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
    --learning_rate 1e-4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1
""" %(input_path[1], output_path[1]))
