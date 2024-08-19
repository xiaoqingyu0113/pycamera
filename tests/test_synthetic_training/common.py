import csv
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from typing import Dict, List, Tuple

def load_csv(file_path:str) -> np.ndarray:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data, dtype=float)



def find_writer_last_step(event_file_path):
        last_step = -1
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                if e.step > last_step:
                    last_step = e.step
        except Exception as e:
            print(f"Failed to read event file {event_file_path}: {str(e)}")
        
        return last_step

def get_summary_writer_path(config):
    logdir = Path(config.model.logdir) 

    if not logdir.exists():
        logdir.mkdir(parents=True)
        ret_path =logdir / 'run00'
    else:
        # get the largest number of run in the logdir using pathlib
        paths = list(logdir.glob('*run*'))
        indices = [int(str(p).split('run')[-1]) for p in paths]

        if len(indices) == 0:
            max_run_num = 0
        else:
            max_run_num = max(indices)

        if config.model.continue_training:
            ret_path =logdir / f'run{max_run_num:02d}'    
        else:
            ret_path =logdir / f'run{1+max_run_num:02d}'

    return ret_path

def get_summary_writer(config) -> Tuple[SummaryWriter, int]:
    '''
        Get the summary writer
    '''
    initial_step = 0
    run_path = get_summary_writer_path(config)
    tb_writer = SummaryWriter(log_dir=run_path)

    # update initial step if continue training
    if config.model.continue_training:
        loss_dir = run_path/'loss_training'
        loss_dir = list(loss_dir.glob('events.out.tfevents.*'))
        initial_step = max([find_writer_last_step(str(rd)) for rd in loss_dir])

    return tb_writer, initial_step