
GPU_ID=0

# For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5

# ACE05
mkdir ace05re_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir ace05  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 32  --save_steps 5000  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16  --seed $seed    \
    --test_file ace05ner_models/PL-Marker-ace05-bert-$seed/ent_pred_test.json  \
    --output_dir ace05re_models/ace05re-bert-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py ace05re ace05re-bert


# SciERC,  --use_ner_results: use the original entity type predicted by NER models
mkdir scire_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  bert_models/scibert-uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16  --seed $seed      \
    --test_file sciner_models/PL-Marker-scierc-scibert-$seed/ent_pred_test.json  \
    --use_ner_results \
    --output_dir scire_models/scire-scibert-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py scire scire-scibert


# ACE04
mkdir ace04re_models
for data_spilt in 0 1 2 3 4; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir ace04  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 384  --max_pair_length 40  --save_steps 2500  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16  --seed 42  \
    --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
    --test_file ace04ner_models/PL-Marker-ace04-bert-$data_spilt/ent_pred_test.json  \
    --output_dir ace04re_models/ace04re-bert-$data_spilt  --overwrite_output_dir
done;
python3 sumup.py ace04re ace04re-bert






