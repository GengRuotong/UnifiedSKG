# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py \
# conda activate py3.7pytorch1.8new
export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning_new
export WANDB_ENTITY=ruotonggeng

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py \
    --run_name expert4_de_phm32_top2gate \
    --local_rank -1 \
    --seed 3407 \
    --cfg Salesforce/T5_base_prefix_summary_5domains_upsample2_res_expert12.cfg \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm True \
    --data_folder_path data/sample_datas_wo_prefix \
    --output_dir output/T5_base_prefix_tuning/5domain_expert8_dephm32_top2gate \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --overwrite_output_dir \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 1 \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --metric_for_best_model avr \
    --greater_is_better true \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --adafactor true \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 128 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 
