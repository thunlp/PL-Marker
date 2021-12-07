
GPU_ID=0

# CoNLL03
mkdir conll03_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner_BIO.py  --model_type roberta  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir conll03  \
    --learning_rate 1e-5  --num_train_epochs 8  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  \
    --train_file eng.train --dev_file eng.testa --test_file eng.testb  \
    --output_dir conll03_models/BIO-conll03-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py conll03ner BIO-conll03-roberta


# OntoNote 5.0
mkdir ontonotes_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner_BIO.py  --model_type roberta  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir ontonotes  \
    --learning_rate 1e-5  --num_train_epochs 4  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 5000  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir ontonotes_models/BIO-ontonotes-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py ontonotesner BIO-ontonotes-roberta


# Few-NERD
mkdir fewnerd_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner_BIO.py  --model_type roberta  \
    --model_name_or_path  bert_models/roberta-large    \
    --data_dir fewnerd  \
    --learning_rate 1e-5  --num_train_epochs 3  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 5000  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  \
    --train_file train.txt --dev_file dev.txt --test_file test.txt  \
    --output_dir fewnerd_models/BIO-fewnerd-roberta-$seed  --overwrite_output_dir
done;
# Average the scores
python3 sumup.py fewnerdner BIO-fewnerd-roberta





