# python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \
# conda activate py3.7pytorch1.8new

export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_ft_w_prefix
export WANDB_ENTITY=ruotonggeng

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py \
    --run_name mt_multi_5domains_wo_prefix \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --data_folder_path data/sample_datas_w_prefix_ahead/ \
    --output_dir output/T5_base_ft_wo_prefix_5domains/multi_domain \
    --seed 2 \
    --cfg Salesforce/T5_base_finetune_summary_5domains_upsample2.cfg \
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
    --eval_steps 2000 \
    --metric_for_best_model avr \
    --greater_is_better true \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --adafactor true \
    --learning_rate 8e-4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 

