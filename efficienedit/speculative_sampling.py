import torch
from tqdm import tqdm
import torch
import re
from efficienedit.kvcache_model import KVCacheModel
from efficienedit.utils import norm_logits, sample, max_fn
from typing import Literal

from transformers import AutoTokenizer


###prefix match###
def find_suffix_and_return_remaining(tensor_a, tensor_b, gamma):
    # Step 1: Check if gamma is valid
    if gamma <= 0 or gamma > len(tensor_a):
        return None  # Invalid gamma
    
    # Extract the suffix of tensor_a
    suffix_a = tensor_a[-gamma:]
    
    # Convert tensors to lists for easier handling (if needed)
    suffix_a_list = suffix_a.tolist()
    tensor_b_list = tensor_b.tolist()
    
    # Step 2: Find the suffix in tensor_b
    len_suffix = len(suffix_a_list)
    len_b = len(tensor_b_list)
    
    for i in range(len_b - len_suffix + 1):
        # Check if the current window matches the suffix
        if tensor_b_list[i:i+len_suffix] == suffix_a_list:
            # Step 3: Return the remaining part of tensor_b
            remaining = tensor_b[i+len_suffix:]
            return remaining
    
    # No match found
    return None

###autoregressive_decoding###
@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, eos_token_id_tensor : torch.Tensor,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + N
    count = 0
    past_key_values = None
    prompt_len = x.shape[1]
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        if temperature == 0 :
            logits = outputs.logits[:, -1, :]
            idx_next = logits.argmax(dim=-1, keepdim=True)            
        else:
            last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
            idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
        current_token = x[0][prompt_len:].flatten()
        eos_tokens = eos_token_id_tensor.flatten()
        if torch.isin(current_token, eos_tokens).any():
            return x 
    return x
###speculative_decoding###
def speculative_sampling_original(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, eos_token_id_tensor : torch.Tensor,
                         max_len : int = 2048 , gamma : int = 7,
                         temperature : float = 0, top_k : int = 0, top_p : float = 0) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    prompt_len =  prefix.shape[1]
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        
        n = prefix_len + gamma - 1
        for i in range(gamma):
            current_top_k_tokens = torch.topk(target_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
            candidate_top_k_tokens =  torch.topk(approx_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
            if candidate_top_k_tokens[0][0] == current_top_k_tokens[0][0]:
                accepted_count += 1
            else:
                n = prefix_len + i - 1
                break

        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            t = torch.argmax(target_model_cache._prob_history[:, n, :], dim=-1).unsqueeze(0)
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = torch.argmax(target_model_cache._prob_history[:, n, :], dim=-1).unsqueeze(0)
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        prefix = torch.cat((prefix, t), dim=1)
        current_token = prefix[0][prompt_len:].flatten()
        eos_tokens = eos_token_id_tensor.flatten()
        if torch.isin(current_token, eos_tokens).any():
            return prefix 
    return prefix

###efficient_edit###
@torch.no_grad()
def efficient_edit_speculative_sampling(prefix : torch.Tensor, precode : torch.Tensor, target_model : torch.nn.Module, draft_model: torch.nn.Module,eos_token_id_tensor: torch.Tensor, 
                         max_len : int , policy, edit_gamma: int=7, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> torch.Tensor:
    
    end_state = False
    # question len
    prompt_len = prefix.shape[1]
    # total max len
    # max_len = 1500
    T = prompt_len + max_len
    # input batch size must be 1
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    device = target_model.device
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    draft_model_cache = KVCacheModel(draft_model, temperature, top_k, top_p)
    
    target_generate = 0
    edit_generate = 0
    
    while prefix.shape[1] <= T :
        # init count #
        draft_accepted_count = 0
        edit_accepted_count = 0 
        if precode.shape[1]>0:
            prefix_len = prefix.shape[1]
            prefix = torch.cat((prefix, precode), dim=1)
            _ = target_model_cache.generate(prefix, 1)
            gamma = precode.shape[1]
            n = prefix_len + gamma - 1
            for i in range(gamma):
                current_top_k_tokens = torch.topk(target_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
                if precode[0][i] == current_top_k_tokens[0][0]:
                    draft_accepted_count += 1
                else:
                    n = prefix_len + i - 1
                    break
            print(draft_accepted_count,precode.shape[1])

            prefix = _ [:, :n+1]
            if n < prefix_len + gamma - 1:
                t = torch.argmax(target_model_cache._prob_history[:, n, :], dim=-1).unsqueeze(0)
                target_model_cache.rollback(n+1)
            else:
                # all approx model decoding accepted
                assert n == target_model_cache._prob_history.shape[1] - 1
                t = torch.argmax(target_model_cache._prob_history[:, n, :], dim=-1).unsqueeze(0)
                target_model_cache.rollback(n+2)
            precode = precode[:,draft_accepted_count:]
            temp_prefix = torch.cat((prefix, t), dim=1)
            current_token = temp_prefix[0][prompt_len:].flatten()
            eos_tokens = eos_token_id_tensor.flatten()
            if torch.isin(current_token, eos_tokens).any():
                end_state = True
            if end_state:
                return temp_prefix

        ### edit postion ###
        edit_gamma = edit_gamma
        current_token = prefix[0][prompt_len:].flatten()
        eos_tokens = eos_token_id_tensor.flatten()

        while prefix.shape[1] <= T:
            edit_accepted_count = 0
            prefix_len = prefix.shape[1]
            x = draft_model_cache.generate(prefix, edit_gamma)
            _ = target_model_cache.generate(x, 1)
            n = prefix_len + edit_gamma - 1

            for i in range(edit_gamma):
                if policy=='greedy':
                    ###greedy###
                    current_top_k_tokens = torch.topk(target_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
                    candidate_top_k_tokens =  torch.topk(draft_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
                    if candidate_top_k_tokens[0][0] == current_top_k_tokens[0][0]:
                        edit_accepted_count += 1
                    else:
                        n = prefix_len + i - 1
                        break
                elif policy=='topk':
                    ###top-k###
                    current_top_k_tokens = torch.topk(target_model_cache._prob_history[:, prefix_len + i - 1, :], 5).indices
                    candidate_top_k_tokens =  torch.topk(draft_model_cache._prob_history[:, prefix_len + i - 1, :], 1).indices
                    if candidate_top_k_tokens[0][0] in current_top_k_tokens[0]:
                        edit_accepted_count += 1
                    else:
                        n = prefix_len + i - 1
                        break
                elif policy=='direct':
                    ###direct###
                    edit_accepted_count += 1
                else:
                    ###entropy###
                    k = 3
                    logits = target_model_cache._prob_history[:, prefix_len + i - 1, :]
                    topk_logits, topk_indices = torch.topk(logits, k)
                    topk_probs = torch.softmax(topk_logits, dim=-1)
                    entropy = -torch.sum(topk_probs * torch.log(topk_probs)).item()
                    normalized_entropy = entropy / torch.log(torch.tensor(k))
                    threshold = int(k * normalized_entropy)
                    if threshold < 1:
                        threshold = 1
                    current_token = x[0][prefix_len + i]

                    if current_token in topk_indices[0][:threshold]:
                        edit_accepted_count += 1
                    else:
                        n = prefix_len + i - 1
                        break

        
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            draft_model_cache.rollback(n+1)

            if n < prefix_len + edit_gamma - 1:
                t = torch.argmax(target_model_cache._prob_history[:, n, :], dim=-1).unsqueeze(0)
                target_model_cache.rollback(n+1)
            else:
                assert n == target_model_cache._prob_history.shape[1] - 1
                t = torch.argmax(target_model_cache._prob_history[:, -1, :], dim=-1).unsqueeze(0)
                target_model_cache.rollback(n+2)
            prefix = torch.cat((prefix, t), dim=1)

            current_token = prefix[0][prompt_len:].flatten()
            eos_tokens = eos_token_id_tensor.flatten()
            if torch.isin(current_token, eos_tokens).any():
                end_state = True
                break

            temp_precode = find_suffix_and_return_remaining(prefix[0],precode[0],edit_gamma)
            if temp_precode!=None:
                precode = temp_precode.unsqueeze(0)
                break

        if end_state:
            return prefix
            
    return prefix
