# T5_base_prefix_summary_3domains_upsample2_embedding.cfg

export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning_explore
export WANDB_ENTITY=ruotonggeng

python train.py \
    --run_name mt_multi_prefix_initial \
    --seed 2 \
    --cfg Salesforce/T5_base_prefix_summary_3domains_upsample2_embedding.cfg \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --freeze_plm True \
    --data_folder_path data/sample_datas_wo_prefix \
    --output_dir output/T5_base_prefix_tuning/multi_domain_prefix_initial/ \
    --no_cuda True \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --num_train_epochs 10 \
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
    --learning_rate 8e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 
