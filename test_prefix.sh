# python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \
# conda activate py3.7pytorch1.8new


export WANDB_API_KEY=3b9858e8352beadda80313599d455c2abfde4ba7
export WANDB_PROJECT=T5_base_prefix_tuning
export WANDB_ENTITY=ruotonggeng

CUDA_VISIBLE_DEVICES=1 python train.py \
    --run_name T5_base_prefix_summary \
    --pretrained_model_path pretrained_model/chinese_t5_pegasus_base/ \
    --domain_name mt_maicai \
    --data_folder_path data/sample_datas_wo_prefix/single_domain/mt_maicai/ \
    --output_dir output/T5_base_prefix_summary/single_domain/mt_maicai/predict_results \
    --load_weights_from output/T5_base_prefix_summary/single_domain/mt_maicai/checkpoint-28000/pytorch_model.bin \
    --seed 2 \
    --cfg Salesforce/T5_base_prefix_summary.cfg \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --generation_num_beams 1 \
    --generation_max_length 128 \
    --input_max_length 512 \
    --num_beams=1 