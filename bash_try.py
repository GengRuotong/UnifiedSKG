import os
import shutil

def mkdir(folder_path):
    # 功能：如果文件夹不存在则创建文件夹，否则删掉原有文件夹创建新文件夹
    folder_path = folder_path.strip().rstrip("//")
    isExists = os.path.exists(folder_path)
    if not isExists:
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

domain_list = ['mt_maoyanyanchu', 'mt_taxi-yonghu', 'mt_maicai', 'mt_waimai', 'mt_youxuan', 'mt_multi']
output_folder = "output/T5_base_prefix_summary/"

for domain_name in domain_list:
    output_path = output_folder + domain_name
    mkdir(output_path)

    os.system("""
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning
export WANDB_ENTITY=mt_prefix_tuning

CUDA_VISIBLE_DEVICES=1 python try.py \
    --run_name T5_base_prefix_summary \
    --seed 2 \
    --cfg Salesforce/T5_base_prefix_summary.cfg \
    --do_train \
    --do_eval \
    --do_predict \
    --domain_name %s \
    --predict_with_generate \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 4 \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --metric_for_best_model avr \
    --greater_is_better true \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --adafactor true \
    --learning_rate 1e-3 \
    --predict_with_generate \
    --output_dir %s \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 
""" %(domain_name, output_path))