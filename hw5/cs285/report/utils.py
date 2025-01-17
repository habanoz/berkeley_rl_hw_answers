import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams, image as mpimg
import numpy as np


def print_seeds(path):
    seed_dirs = sorted(glob.glob(path + "*"))

    n_paths = len(seed_dirs)

    for p in range(n_paths):
        print(seed_dirs[p])


def average_metrics(path, stats_idx, shift=0, clip=0):
    seed_dirs = sorted(glob.glob(path + "*" + "/eval.txt"))
    n_paths = len(seed_dirs)

    sum = None
    for p in range(n_paths):
        seed_dir = seed_dirs[p]
        seed_arr = np.loadtxt(seed_dir)
        c = len(seed_arr[stats_idx]) if clip == 0 else clip
        if sum is None:
            sum = seed_arr[stats_idx][shift:c]
        else:
            sum += seed_arr[stats_idx][shift:c]
    avg = sum / n_paths
    return avg


def plot_arrays(title, arrays, labels):
    plt.clf()
    rcParams['figure.figsize'] = 24, 12

    plt.title(title)

    for a in range(len(arrays)):
        array = arrays[a]
        label = labels[a]

        plt.plot(array, label=label)

    plt.legend()
    plt.grid()
    plt.show()


def plot_metrics_from_path_per_seed(title, path, stats_idx, shift=0, clip=0):
    plt.clf()
    mult = 8
    seed_dirs = sorted(glob.glob(path + "*" + "/eval.txt"))

    n_paths = len(seed_dirs)

    rcParams['figure.figsize'] = mult * n_paths, mult
    plt.title(title)
    for p in range(n_paths):
        seed_dir = seed_dirs[p]
        seed_arr = np.loadtxt(seed_dir)
        c = len(seed_arr[stats_idx]) if clip == 0 else clip
        plt.plot(seed_arr[stats_idx][shift:c], label=f'Seed {p}')

    plt.legend()
    plt.grid()
    plt.show()


def plot_metrics_from_path_avg(title, path, stats_idx, shift=0, clip=0):
    plt.clf()
    mult = 8
    seed_dirs = sorted(glob.glob(path + "*" + "/eval.txt"))

    n_paths = len(seed_dirs)

    rcParams['figure.figsize'] = 2 * mult, mult
    plt.title(title)

    sum = None
    for p in range(n_paths):
        seed_dir = seed_dirs[p]
        seed_arr = np.loadtxt(seed_dir)
        c = len(seed_arr[stats_idx]) if clip == 0 else clip
        if sum is None:
            sum = seed_arr[stats_idx][shift:c]
        else:
            sum += seed_arr[stats_idx][shift:c]
    avg = sum / n_paths

    plt.plot(avg)

    plt.grid()
    plt.show()


def show_graphs_seed_steps(path, sub_path_spec, steps, title=None):
    seed_dirs = sorted(glob.glob(path + "*"))
    seed_dirs = [dir for dir in seed_dirs if Path(dir + "/eval.txt").exists()]

    n_steps = len(steps)
    n_paths = len(seed_dirs)
    mult = 8

    rcParams['figure.figsize'] = mult * n_paths, mult * n_steps

    fig, ax = plt.subplots(n_steps, n_paths, squeeze=False)
    if title is not None:
        fig.suptitle(title)

    for p in range(n_paths):
        for s in range(n_steps):
            seed_dir = seed_dirs[p]
            step = steps[s]
            img = mpimg.imread(seed_dir + f'/curr_{sub_path_spec}/{sub_path_spec}{step}.png')
            ax[s, p].imshow(img)

    for p in range(n_paths):
        ax[0, p].set_title(f'{Path(seed_dirs[p]).stem[9:50]}')
    for s in range(n_steps):
        ax[s, 0].set_ylabel(f'{steps[s]:,}')

    fig.tight_layout()

def merge_traj(path, template):
    files = sorted(glob.glob(path + "/*.png"))
    images = np.array([(mpimg.imread(f)) for f in files])

    imf_ = np.sum(images, axis=0) - template * len(images)

    b0 = imf_[:, :, 1]
    b0 = b0 - b0.min()
    b0 = b0 / b0.max()
    b0[b0 > 0.95] = 1.0

    # plt.imshow(b0, cmap='gray')
    return b0


def plot_merged_traj_seeds(path, grp):
    seed_paths = sorted(glob.glob(path + f"/{grp}/expl"))

    f, ax = plt.subplots(2, 2)

    for p in range(len(seed_paths)):
        spath = seed_paths[p]

        template = mpimg.imread(spath+"/../../expl_last_traj.png")
        im = merge_traj(spath, template)

        ax[p // 2, p % 2].imshow(im)
    plt.show()


def plot_merged_traj_seeds_1row(path, grp):
    seed_paths = sorted(glob.glob(path + f"/{grp}/expl"))

    f, ax = plt.subplots(1, len(seed_paths))

    for p in range(len(seed_paths)):
        spath = seed_paths[p]

        template = mpimg.imread(spath+"/../../expl_last_traj.png")
        im = merge_traj(spath, template)

        ax[p].imshow(im)
    plt.show()