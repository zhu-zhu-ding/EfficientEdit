ROOT=/EfficientEdit/data/CodeIF-Bench/L2_test

python run_pass@k.py \
    --output_file $ROOT/result.jsonl \
    --log_file $ROOT/log.jsonl \
    --source_code_root $ROOT/Source_Code \
    --data_file $ROOT/data.jsonl \
    --n 1 \
    --k 1