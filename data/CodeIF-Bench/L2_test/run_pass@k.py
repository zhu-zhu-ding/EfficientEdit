from pathlib import Path
import json
import subprocess
import psutil
from subprocess import run
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
import textwrap
from func_timeout import func_set_timeout
import func_timeout
from util import (
    read_json,
    save_json
)
import re
import copy
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--log_file', type=Path)
    parser.add_argument('--source_code_root', type=Path, default=Path('$ROOT/Source_Code'))
    parser.add_argument('--data_file', type=Path, default=Path('$ROOT/data.jsonl')) # data.jsonl
    parser.add_argument('--k', type=str, default='1') # k in pass_at_k
    parser.add_argument('--n', type=int, default=1) # number of completions per task
    return parser.parse_args()


def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

import subprocess
import os
import psutil
import time

@func_set_timeout(60)
def execution_tests(args, data):
    project_path = os.path.join(args.source_code_root, data['project_path'])
    command = ['python', 'setup.py', 'pytest', '--addopts']
    for test in data['tests']:
        process = subprocess.Popen(command + [test], cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024:  # 5GB memory usage per test
                    process.terminate()
                    process.wait()
                    return 'OOM'  # Out of Memory
                return_code = process.poll()
                if return_code is not None:
                    stdout, stderr = process.communicate()
                    if return_code != 0:
                        # If there is an error, check if it's a memory issue or other issues
                        process.terminate()
                        process.wait() 
                        if 'MemoryError' in stderr:  # Check for MemoryError specifically
                            return f'Error: {test}'
                        else:
                            return f'Error: {test}'
                    else:
                        break
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
            return 'Timeout Error'  # Timeout Error
        except psutil.NoSuchProcess:
            process.terminate()
            process.wait()
            return 'Process Not Found'  # Process not found error
        except Exception as e:
            process.terminate()
            process.wait()
            return f'Other Error: {str(e)}'  # Other Error with details
        finally:
            process.terminate()
            process.wait()
    return 'Pass'  # Pass

def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))


def SetUp_evaluation(args, data, completion):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])

    # rename the original completion file as tmp_completion
    run(['cp', completion_path, completion_tmp_path])

    # write the new completion file
    sos, eos = data['signature_position'][0]-1, data['body_position'][1]
    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    with open(completion_path, 'w') as f:
        f.write(''.join(file_lines))


def TearDown_evaluation(args, data):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['mv', completion_tmp_path, completion_path])


def check_correctness(args, data):
    completion = data['completions']
    # print(data['completion'])
    if completion == "    pass\n":
        return 'Error'
    completion = adjust_indent(completion, data['indent'])
    
    SetUp_evaluation(args, data, completion)
    try:
        flag = execution_tests(args, data)
    except func_timeout.exceptions.FunctionTimedOut:
        flag = 'TimeOut'
    TearDown_evaluation(args, data)
    return flag


def report_results(args, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError(f'{args.log_file} does not exist')
    
    # Collect passed completions for each namespace
    passed_completion = {}
    with open(args.log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            if 'pass' in js:
                js['Result'] = js['pass']
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completions']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    # Iterate through all completions and count the number of passed completions for each namespace
    results = {}
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completions']
            if namespace not in benchmark_data:
                continue
            if namespace not in results:
                results[namespace] = 0
            if namespace in passed_completion and completion in passed_completion[namespace]:
                results[namespace] += 1
            
    # Compute Pass@k
    k_list = [int(k) for k in args.k.split(',')]
    for k in k_list:
        if k > args.n:
            continue
        pass_at_k = np.mean([compute_pass_at_k(args.n, pass_num, k) for namespace, pass_num in results.items()])
        print(f'pass_at_{k}: {pass_at_k*100}%')


def load_finished_data(args):
    finished_data = {}
    if os.path.exists(args.log_file):
        with open(args.log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                namespace, completion = js['namespace'], js['completions']
                if namespace not in finished_data:
                    finished_data[namespace] = set()
                finished_data[namespace].add(completion)
    return finished_data

def extract_python(code):
    pattern = r"(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return code

def main(args):
    finished_data = load_finished_data(args)
    print(len(finished_data))
    todo_output_data = []
    output_list = read_json(args.output_file,True)
    for js in output_list:
        namespace, completion = js['namespace'], extract_python(js['completions'][0])
        if namespace not in finished_data:
            todo_output_data.append(js)
            finished_data[namespace] = set()
            finished_data[namespace].add(completion)
        # elif completion not in finished_data[namespace]: 
        else:
            todo_output_data.append(js)
            finished_data[namespace].add(completion)         
    del finished_data
    print("TODO Completions: ", len(todo_output_data))

    benchmark_data = {}
    data_list = read_json(args.data_file,True)
    for js in data_list:
        namespace = js['namespace']
        benchmark_data[namespace] = js
    flag = todo_output_data[0]['namespace']
    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['namespace'] in benchmark_data:
                # data = copy.deepcopy(benchmark_data[output['namespace']])
                data = benchmark_data[output['namespace']]
                # print(data)
                data['completions'] =  extract_python(output['completions'][0])
                # output['completions'] =  extract_python(output['code_before'])
                # data['completions'] =  output['code_before']
                print(data['completions'])
                if output['ins_type']=="Functionality Extension":
                    data['tests'] = output["re_tests"]
                else:
                    data["tests"] = output["base_tests"]+ output["re_tests"]
                # data["tests"] = output["base_tests"]
                print(data['tests'])
                flag = check_correctness(args, data)
                output['Result'] = flag
                f.write(json.dumps(output) + '\n')
                f.flush()

    # report_results(args, benchmark_data)


def test_ground_truth(args):
    data = open(args.data_file, 'r').readlines()
    output_f = open('failed_samples.jsonl', 'w')

    for line in tqdm(data):
        js = json.loads(line)
        tests = set(js['tests'])
        js['tests'] = list(tests)
        try:
            flag = execution_tests(args, js)
        except func_timeout.exceptions.FunctionTimedOut:
            flag = 'TimeOut'
        if flag != 'Pass':
            print(js['namespace'])
            output_f.write(json.dumps(js) + '\n')


if __name__ == '__main__':
    args = get_parser()
    if args.output_file is None:
        test_ground_truth(args)
    else:
        main(args)