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
    I = []
    X = []
    Y = []
    T = []

    mrm = []
    erm = []

    time0 = None
    for e in summary_iterator(file):
        if time0 is None:
            time0 = e.wall_time
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                I.append(v.simple_value)
            if v.tag == 'Train_AverageReturn':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
                T.append(int(e.wall_time - time0))
                time0 = e.wall_time
            elif v.tag == 'Mixed_Reward_Mean':
                mrm.append(v.simple_value)
            elif v.tag == 'Env_Reward_Mean':
                erm.append(v.simple_value)
    return I, X, Y, mrm, erm


def read_log_dir(logdir):
    eventfile = glob.glob(logdir + "/events*")[0]
    I, X, Y, mrm, erm = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Mean reward(100): {} | Eval Return: {}'.format(i, x, y))

    n_its = len(I)
    X = [0]*(n_its-len(X))+X
    Y = [0]*(n_its-len(Y))+Y
    mrm = [0]*(n_its-len(mrm))+mrm
    erm = [0]*(n_its-len(erm))+erm

    np.savetxt(logdir + "/eval.txt", np.array([I, X, Y, mrm, erm]))

    return 1 if len(X) > 0 else 0


if __name__ == '__main__':
    log_dirs = glob.glob("data/awac/hw5_expl_q4_awac_medium_supervised_*")

    sum = 0
    for logdir in log_dirs:
        sum += read_log_dir(logdir)

    print(f"{sum} files created")
