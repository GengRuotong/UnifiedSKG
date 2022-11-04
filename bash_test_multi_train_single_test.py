import os
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \

def mkdir(folder_path):
    # 功能：如果文件夹不存在则创建文件夹，否则删掉原有文件夹创建新文件夹
    folder_path = folder_path.strip().rstrip("//")
    isExists = os.path.exists(folder_path)
    if not isExists:
        os.makedirs(folder_path)
    else:
        print("Folder exists for %s" %(folder_path))

domain_list = ['mt_maoyanyanchu', 'mt_taxi-yonghu', 'mt_maicai', 'mt_waimai', 'mt_youxuan', 'mt_multi']
input_folder = "data/sample_datas_wo_prefix/"
output_folder = "output/T5_base_prefix_summary_upsample2/multi_train_single_test_161000/"

for domain_name in domain_list:
    input_path = input_folder + domain_name + '/'
    output_path = output_folder + domain_name
    mkdir(output_path)

    os.system("""
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning
export WANDB_ENTITY=ruotonggeng

CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --run_name T5_base_prefix_tuning \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm True \
    --domain_name %s \
    --data_folder_path %s \
    --output_dir %s \
    --seed 2 \
    --cfg Salesforce/T5_base_prefix_summary.cfg \
    --load_weights_from output/T5_base_prefix_summary_upsample2/mt_all_tasks_upsample2/checkpoint-161000/pytorch_model.bin \
    --do_predict \
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
    --learning_rate 1e-4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 
""" %(domain_name, input_path, output_path))