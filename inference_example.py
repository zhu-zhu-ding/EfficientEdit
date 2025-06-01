import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4.5"
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from  tqdm import tqdm
from util import (
    read_json,
    save_json
)
from peft import PeftModel, LoraConfig, get_peft_model
import re
torch.manual_seed(520)
from argparse import ArgumentParser
from pathlib import Path
from efficienedit.speculative_sampling import autoregressive_sampling,speculative_sampling_original,efficient_edit_speculative_sampling
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--draft_lora_path', type=Path, default=None)
    parser.add_argument('--draft_model', type=Path)
    parser.add_argument('--target_model', type=Path)
    return parser.parse_args()

def code_edit_prompt(instruction,code_before):
    prompt = f"""###User:
You are an expert code editor. Please modify the given ##Code File according to the ##Instruction provided. Provide the complete revised code file with all modifications implemented.
##Instruction
{instruction}
##Code File
```python
{code_before}
```
###Assistant
##Code File
```python
"""
    return prompt

def speculative_sampling_inference(target_model, draft_model, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5):
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = speculative_sampling_original(
                prefix = input_ids,
                approx_model = draft_model,
                target_model = target_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500,
                temperature = temperature, 
                top_k= top_k, 
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}

def efficient_edit_inference(target_model, draft_model, code_before, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5):
    precode = tokenizer.encode(code_before, add_special_tokens=False, return_tensors="pt").to(target_model.device)
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = efficient_edit_speculative_sampling(
                prefix = input_ids, 
                precode = precode, 
                target_model = target_model,
                draft_model = draft_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500 ,
                policy = "greedy",
                temperature = temperature, 
                top_k= top_k, 
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}
def autoregressive_inference(model, input_ids, max_token, eos_token_id_tensor, temperature= 0.2, top_p = 0.95,top_k = 5):
    with contexttimer.Timer() as t:
        with torch.no_grad():
            ####kv cache###
            outputs = autoregressive_sampling(
                x = input_ids, 
                model = model,
                N = max_token ,
                eos_token_id_tensor = eos_token_id_tensor,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p)

    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}

if __name__ == '__main__':
    args = get_parser()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,torch_dtype=torch.float16,device_map="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.float16,device_map="auto")

    if 'Qwen' in str(args.target_model):
        eos_token_id_tensor = torch.tensor([84274,73594,9902,13874,41233,54275,151645]).to(target_model.device)
    else:
        eos_token_id_tensor = torch.tensor([10252,32021]).to(target_model.device)

    if args.draft_lora_path:
        print("load lora")
        draft_model = PeftModel.from_pretrained(draft_model, args.draft_lora_path)
        draft_model = draft_model.merge_and_unload()
        print("load lora end")
    draft_model.eval()
    target_model.eval()

    data = read_json(args.data_file,False)
    result = []

    data = data[:1]
    for item in tqdm(data):
        ####canitedit####
        prompt = code_edit_prompt(item['instruction_lazy'],item['before'])
        code_before = item['before']+'\n'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)

        ###AR###
        # _ = autoregressive_inference(target_model, input_ids,2048, eos_token_id_tensor, temperature=0)
        ###SD###
        # _ = speculative_sampling_inference(target_model = target_model,draft_model = draft_model, eos_token_id_tensor = eos_token_id_tensor, input_ids = input_ids,max_token = 4096, temperature=0)
        ###efficientedit###
        # _ = efficient_edit_inference(target_model = target_model,draft_model = draft_model, code_before= code_before, eos_token_id_tensor = eos_token_id_tensor, input_ids = input_ids,max_token = 4096, temperature=0)


        item['completions'] = [_['result']]
        item['time'] = _['time']
        item['tokens'] = _['tokens']
        item['rate'] = _['rate']
        # item['draft_rate'] = _['draft_rate']
        if 'edit_eval' in str(args.data_file):
            item['output'] = [_['result']]
        result.append(item)
        save_json(args.output_file,result)
