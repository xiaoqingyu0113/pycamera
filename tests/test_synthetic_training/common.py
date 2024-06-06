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

def get_summary_writer(config) -> Tuple[SummaryWriter, int]:
    '''
        Get the summary writer
    '''
    def find_last_step(event_file_path):
        last_step = -1
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                if e.step > last_step:
                    last_step = e.step
        except Exception as e:
            print(f"Failed to read event file {event_file_path}: {str(e)}")
        
        return last_step
    
    logdir = Path(config.model.logdir) / config.dataset.name
    initial_step = 0
    if not logdir.exists():
        logdir.mkdir(parents=True)
        tb_writer = SummaryWriter(log_dir=logdir / 'run00')
    else:
        # get the largest number of run in the logdir using pathlib
        paths = list(logdir.glob('*run*'))
        indices = [int(str(p).split('run')[-1]) for p in paths]

        if len(indices) == 0:
            max_run_num = 0
        else:
            max_run_num = max(indices)
        if config.model.continue_training:
            tb_writer = SummaryWriter(log_dir=logdir / f'run{max_run_num:02d}')
            rundir = logdir/f'run{max_run_num:02d}'/'loss_training'
            rundir = list(rundir.glob('events.out.tfevents.*'))
            initial_step = max([find_last_step(str(rd)) for rd in rundir])
        else:
            tb_writer = SummaryWriter(log_dir=logdir / f'run{1+max_run_num:02d}')
    return tb_writer, initial_step