import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

IGNORE_INDEX = -100
# EOT_TOKEN = "</s>"
# EOT_TOKEN = "<|EOT|>"
EOT_TOKEN = "<|im_end|>"
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

def build_instruction_prompt(instruction: str):
    return instruction

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], masks:Sequence[List[str]], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    for idx, (label, source_len, target, mask_pieces) in enumerate(zip(
        labels, 
        sources_tokenized["input_ids_lens"], 
        targets,
        masks
    )):
        label[:-3] = IGNORE_INDEX
        target_tokens = tokenizer.encode(target,add_special_tokens=False)
        
        for mask_piece in mask_pieces:
            mask_tokens = tokenizer.encode(mask_piece,add_special_tokens=False)
            
            for i in range(len(target_tokens) - len(mask_tokens) + 1):
                if target_tokens[i:i+len(mask_tokens)] == mask_tokens:
                    for j in range(len(mask_tokens)):
                        actual_pos = source_len + i + j
                        if actual_pos < len(label):
                            label[actual_pos] = input_ids[idx][actual_pos]
    
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in labels], batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def train_tokenize_function(examples, tokenizer):
    sources = [build_instruction_prompt(instruction) for instruction in examples['prompt']]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['code']]
    masks = [mask for mask in examples['mask']]
    data_dict = preprocess(sources, targets, masks, tokenizer)
    return data_dict


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)


    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
        print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")
    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
