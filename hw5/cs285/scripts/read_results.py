import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import tensorboard as tb

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    E = []
    T = []
    time0 = None
    for e in summary_iterator(file):
        if time0 is None:
            time0 = e.wall_time
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
                T.append(int(e.wall_time - time0))
                time0 = e.wall_time
            elif v.tag == 'Eval_StdReturn':
                E.append(v.simple_value)
    return X, Y


def read_log_dir(logdir):
    eventfile = glob.glob(logdir + "/events*")[0]
    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Mean reward(100): {} | Eval Return: {}'.format(i, x, y))
    min_size = min(len(X), len(Y))
    np.savetxt(logdir + "/eval.txt", np.array([X[:min_size], Y[:min_size]]))


if __name__ == '__main__':
    log_dirs = glob.glob("data/hw5_expl_q1_env1_rnd_*")

    for logdir in log_dirs:
        read_log_dir(logdir)
