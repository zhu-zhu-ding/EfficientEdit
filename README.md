# EfficientEdit

## About

EfficientEdit is an innovative inference acceleration technique that leverages edit-oriented speculative decoding to accelerate code editing tasks. Demonstrating remarkable performance gains, the method achieves up to 8× speedup on Qwen2.5-Coder-32-Instruct and 13× acceleration on DeepSeek-Coder-33B-Instruct models.

## Install

### Install EfficientEdit

```bash
cd EfficientEdit
conda create --name efficientedit
conda activate efficientedit
pip install requirements
```

### Install CanItEdit

Create your conda environment and notice python>=3.10

```
cd EfficientEdit/data/CanItEdit
conda create --name canitedit python==3.10.16
conda activate canitedit
pip install requirements
```

### Install CodeIF-Bench

```
cd EfficientEdit/data/CodeIF-Bench
conda create --name codeif
conda activate codeif
pip install requirements
```

## Inference

The inference script is inference.sh. Please fill in the required root directory.

```
python inference_example.py \
    --data_file $DATA_DIR/test.jsonl \
    --output_file $RESULTS_DIR/example.jsonl \
    --target_model $MODEL_DIR/target_model \
    --draft_model $MODEL_DIR/draft_model
```

## Fine-Tuning

The fine-tuning script is train.sh. Please fill in the required root directory.

```
DATA_PATH=""
OUTPUT_PATH=""
MODEL_PATH=""

export CUDA_VISIBLE_DEVICES=1,2
deepspeed --num_gpus 2 --master_port 6001 mask_lora.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed ./configs/ds_config_zero3.json \
    --bf16 True
```

You can download the lora checkpoints at [link](https://figshare.com/s/9b3f68f054bc73936f11).

## Test

### CanItEdit

You can run data/CanItEdit/test.py to compute the Pass@k of CanItEdit.

### CodeIF-L1_test

You can run data/CodeIF-Bench/L1_test/test.py to compute the Pass@k of L1.

### CodeIF-L2_test

You can run data/CodeIF-Bench/L2_test/run_pass_k.sh to compute the Pass@k of L2. The script is shown below.

```python
ROOT=/EfficientEdit/data/CodeIF-Bench/L2_test
python run_pass@k.py \
    --output_file $ROOT/result.jsonl \
    --log_file $ROOT/log.jsonl \
    --source_code_root $ROOT/Source_Code \
    --data_file $ROOT/data.jsonl \
    --n 1 \
    --k 1
```
The arguments are explained as follows.

+ `output_file`: the model's predictions.
+ `log_file`: the output file that stores the evaluation results.
+ `source_code_root`: the path of repositories. The original repositories can be downloaded from [link](https://figshare.com/s/aa2ec81006727d9ddb0c).
+ `data_file`: the metadata file.
+ `n`: number of completions per task, e.g., `1`
+ `k`:the k value in Pass@k, e.g., `1`

