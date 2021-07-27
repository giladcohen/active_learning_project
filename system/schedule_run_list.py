import torch
from time import time, sleep
import subprocess as sp
import os
import yaml

def run_cmd(cmd):
    print('start running command: \n{}'.format(cmd))
    process = sp.call(cmd, shell=True)
    print('finished running command: \n{}'.format(cmd))

# buffer time for running different commands on the same GPU
SAFE_TIME = 40
ALLOWED_GPUS_INDS = [0,1,2,3,4,5,6,7]
# yml_command_file = os.path.join(os.path.dirname(__file__), 'commands.yml')
#debug:
yml_command_file = 'active_learning_project/system/commands.yml'

with open(yml_command_file) as f:
    d = yaml.load(f.read())
# list of (command, required memory in GB). List the command by priority. The script will always try to clear up the
# top of the list first
COMMANDS = d['commands']

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
        if required_mem <= mem and is_safe_gpu(gpu_id) and gpu_id in ALLOWED_GPUS_INDS:
            return gpu_id

    return -1

def main():
    start_time = time()
    while len(COMMANDS):
        gpu_id = wait_for_idle_gpu()
        if gpu_id == -1:
            print('No GPU with a free memory of {} GB to run command: \n{}'.format(COMMANDS[0][1], COMMANDS[0][0]) +
            '\nAutomatic retry in 60 seconds.')
            sleep(60)  # sample again in 60 seconds
        else:
            last_time_used[gpu_id] = time()
            command = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id) + COMMANDS.pop(0)[0]
            new_pid = os.fork()
            if new_pid == 0:
                pid = os.getpid()
                run_cmd(command)
                try:  # might return an error
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    # It was already killed
                    pass
                break
            sleep(5)

    if len(COMMANDS) == 0:  # if we got here after the loop actually finished
        print('Done running all commands. It took {} seconds.'.format(time() - start_time))


if __name__ == '__main__':
    main()
