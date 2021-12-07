
GPU_ID=0

# We can use larger max_pair_length for evaluation to process the larger number of spans in test set with a faster speed, e.g. 1800 for CoNLL03

# CoNLL03
mkdir conll03_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type robertaspan  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir conll03  \
    --learning_rate 1e-5  --num_train_epochs 8  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 1000  --max_mention_ori_length 8  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  \
    --train_file eng.train --dev_file eng.testa --test_file eng.testb  \
    --output_dir conll03_models/TokenCat-conll03-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py conll03ner TokenCat-conll03-roberta


# OntoNote 5.0
mkdir ontonotes_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type robertaspan  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir ontonotes  \
    --learning_rate 1e-5  --num_train_epochs 4  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 5000  --max_pair_length 3240  --max_mention_ori_length 16    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir ontonotes_models/TokenCat-ontonotes-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py ontonotesner TokenCat-ontonotes-roberta


# Few-NERD
mkdir fewnerd_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type robertaspan  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir fewnerd  \
    --learning_rate 1e-5  --num_train_epochs 3  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 5000  --max_pair_length 6440  --max_mention_ori_length 16    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  \
    --train_file train.txt --dev_file dev.txt --test_file test.txt  \
    --output_dir fewnerd_models/TokenCat-fewnerd-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py fewnerdner TokenCat-fewnerd-roberta










