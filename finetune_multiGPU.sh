export base_model='VietAI/gpt-neo-1.3B-vietnamese-news'
export data_path='./GenMedGPT-5k.json'
export output_dir='./lora-chatdoctor-5k'
export load_in_8bit=True

export seed=42
export batch_size=128
export micro_batch_size=4
export num_epochs=3
export learning_rate=3e-4
export cutoff_len=256
export val_set_size=500
export warmup_steps=10
export logging_steps=5
export eval_steps=10
export save_steps=10
export save_total_limit=3

export lora_r=8
export lora_alpha=16
export lora_dropout=0.05
export lora_target_modules='[q_proj, v_proj]'

export train_on_inputs=True
export group_by_length=False

export resume_from_checkpoint=None

export WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun --nproc_per_node=8 --master_port=1234 
    finetune.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \
    --load_in_8bit $load_in_8bit \
    --seed $seed \
    --batch_size $batch_size \
    --micro_batch_size $micro_batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --cutoff_len $cutoff_len \
    --val_set_size $val_set_size \
    --warmup_steps $warmup_steps \
    --logging_steps $logging_steps \
    --eval_steps $eval_steps \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_target_modules $lora_target_modules \
    --train_on_inputs $train_on_inputs \
    --group_by_length $group_by_length \
    --resume_from_checkpoint $resume_from_checkpoint
