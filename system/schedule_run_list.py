import torch
from time import time, sleep
import subprocess as sp
import os

def run_cmd(cmd):
    print('start running command: \n{}'.format(cmd))
    process = sp.call(cmd, shell=True)
    print('finished running command: \n{}'.format(cmd))

# buffer time for running different commands on the same GPU
SAFE_TIME = 40

# tuple of (command, required memory in GB). List the command by priority. The script will always try to clear up the
# top of the list first
COMMANDS = [
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_02 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_03 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_04 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_05 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_06 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_07 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_08 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),
    ('python active_learning_project/scripts/train.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_09 '
     '--dataset svhn '
     '--net resnet101 '
     '--epochs 200 '
     '--patience 2 '
     '--cooldown 1', 10000),

    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack fgsm '
     '--targeted True', 10000),
    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack pgd '
     '--targeted True', 10000),
    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack jsma '
     '--targeted True', 10000),
    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack cw '
     '--targeted True', 10000),
    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack deepfool '
     '--targeted False', 10000),

    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/svhn/resnet101/resnet101_00 '
     '--attack ead '
     '--targeted False '
     'batch_size 50', 10000),

    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/cifar100/resnet34/adv_robust '
     '--attack ead '
     '--targeted False', 10000),
]

WORKING_DIR = '/home/gilad/python3_workspace'
os.chdir(WORKING_DIR)

# Whenever a GPU is used, we documented the timestamp here and compare it later with SAFE_TIME
last_time_used = {}

assert torch.cuda.is_available(), 'The file ' +  __file__ + ' is designed to run only with cuda. exiting.'

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def is_safe_gpu(gpu_id):
    last_call_timestamp = last_time_used.get(gpu_id, None)
    if last_call_timestamp is None:
        return True
    else:
        return (time() - last_call_timestamp) > SAFE_TIME

def wait_for_idle_gpu():
    # always prioritize the top of the COMMAND list
    required_mem = COMMANDS[0][1]

    memory_free_list = get_gpu_memory()
    for gpu_id, mem in enumerate(memory_free_list):
        if required_mem <= mem and is_safe_gpu(gpu_id):
            return gpu_id

    return -1

def main():
    start_time = time()
    while len(COMMANDS):
        gpu_id = wait_for_idle_gpu()
        if gpu_id == -1:
            print('No GPU with a free memory of {} GB to run command: \n{}'.format(COMMANDS[0][1], COMMANDS[0][0]) +
            '\nAutomatic retry in 1 minute.')
            sleep(60)  # sample again in 1 minute
        else:
            last_time_used[gpu_id] = time()
            command = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id) + COMMANDS.pop(0)[0]
            pid = os.fork()
            if pid == 0:
                run_cmd(command)
                # os.waitpid(pid, os.WNOHANG)
                exit(0)
            sleep(5)

    if len(COMMANDS) == 0:  # if we got here after the loop actually finished, and no
        print('Done running all commands. It took {} seconds.'.format(time() - start_time))


if __name__ == '__main__':
    main()
