#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # One cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=23:59:59 # Run for 3 days, at most
#SBATCH --job-name=ECG_MIT # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address
#SBATCH --output=outputs/ecg_mit.out
#SBATCH --error=outputs/ecg_mit.err
#SBATCH --constraint=gpu80

# seq_len=3200
model=CustomLSTM #TEMPO #PatchTST #_multi
electri_multiplier=1
traffic_multiplier=1

for seq_len in 720
do
for percent in 100 #5 10
do
for pred_len in 1 #96 #192 336 720 #96 #720 #336 #192 #336 #720 #96 #720 #96 #96 #336 #192 #96 #336 96 # 96 192
do
for tmax in 20
do
for lr in 0.001 #0.005 0.000005 
do
for gpt_layer in 6 #3 #6 #6 #3 #6 #0 
do
for equal in 1 #0
do
for prompt in 1 #0 #1 #0
do
mkdir -p logs/$model
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ecg_mit_no_pool_$model'_'$gpt_layer
echo logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ecg_mit_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log



python main_multi_6domain_release.py \
    --datasets ecg_mit \
    --target_data ecg_mit \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./lora_revin_6domain_checkpoints'_'$prompt/ \
    --model_id ECG_MIT_TEMPO'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 256 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 1 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --tmax $tmax \
    --cos 1 \
    --is_gpt 1 #>> logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ettm2_pmt1_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log

done
done
done
done
done
done
done
done
