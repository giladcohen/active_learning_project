import torch
import time
import subprocess as sp
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = sp.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))

COMMANDS = [
    ('python active_learning_project/scripts/attack.py '
    '--checkpoint_dir /data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
    '--attack fgsm'
    '--targeted True'
    '--attack_dir run_example1', 3000),

    ('python active_learning_project/scripts/attack.py '
    '--checkpoint_dir /data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
    '--attack fgsm'
    '--targeted True'
    '--attack_dir run_example2', 4000),

    ('python active_learning_project/scripts/attack.py '
    '--checkpoint_dir /data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
    '--attack fgsm'
    '--targeted True'
    '--attack_dir run_example3', 10000),

    ('python active_learning_project/scripts/attack.py '
     '--checkpoint_dir /data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
     '--attack fgsm'
     '--targeted True'
     '--attack_dir run_example4', 1000),
]

WORKING_DIR = '/home/gilad/python3_workspace'
os.chdir(WORKING_DIR)

assert torch.cuda.is_available(), 'The file ' +  __file__ + ' is designed to run only with cuda. exiting.'

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def wait_for_idle_gpu():
    # always try to prioritize the top of the COMMAND list
    required_mem = COMMANDS[0][1]

    memory_free_list = get_gpu_memory()
    for gpu_id, mem in enumerate(memory_free_list):
        if required_mem <= mem:
            return gpu_id

    return -1


while len(COMMANDS):
    gpu_id = wait_for_idle_gpu()
    if gpu_id == -1:
        time.sleep(30)
    else:
        command = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id) + COMMANDS.pop()[0]
        run_cmd(command)
        time.sleep(5)

print('Done running all commands.')
