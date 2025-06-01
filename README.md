# EfficientEdit

## About

EfficientEdit is an innovative inference acceleration technique that leverages edit-oriented speculative decoding to accelerate code editing tasks. Demonstrating remarkable performance gains, the method achieves up to 8× speedup on Qwen2.5-Coder-32-Instruct and 13× acceleration on DeepSeek-Coder-33B-Instruct models.

## Install

### Install EfficientEdit

`cd EfficientEdit `

`pip install requirements`

### Install CanItEdit

`cd data/CanItEdit`

Create your conda environment and notice python>=3.10

``pip install requirements``

Install CodeIF-Bench

`cd data/CodeIF-Bench`

Create your conda environment

``pip install requirements``

## Test

### L2_test

Users can run data/CodeIF-Bench/L2_test/run_pass_k.sh to compute the Pass@k of L2. The script is shown below.

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

## Inference

The inference script is [inference.sh](https://github.com/zhu-zhu-ding/EfficientEdit/blob/main/inference.sh)

## Fine-Tuning

The fine-tuning script is [train.sh](https://github.com/zhu-zhu-ding/EfficientEdit/blob/main/fine-tuning/train.sh)
The lora checkpoint is [link](https://figshare.com/s/9b3f68f054bc73936f11)

