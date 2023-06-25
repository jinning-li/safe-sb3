from jupyterlab_h5web import H5Web
import pickle
import matplotlib.pyplot as plt
import numpy as np

# what about the data from h5py
import h5py

def visualize_h5py(args):
    # Specify the filename of the h5py file
    h5py_filename = args['h5py_path']

    hf = h5py.File(h5py_filename, 'r')
    # Get a list of dataset names
    dataset_names = list(hf.keys())

    print("Names of h5py datasets:")
    for name in dataset_names:
        print(name)

    re = np.array(hf["reward"])
    cost = np.array(hf["cost"])
    ac = np.array(hf["action"])
    plt.figure()
    # Plot time versus acceleration
    plt.plot(range(len(re)), re, label = "reward")
    plt.plot(range(len(cost)), cost, label = "cost")
    plt.legend()
    plt.xlabel('TimeStamp')
    plt.ylabel('Reward/cost')
    plt.title('TimeStamp vs. Reward/cost')
    plt.figure()
    # Plot time versus acceleration
    plt.plot(np.arange(ac.shape[0]), ac[:,0], label = 'lat')
    plt.plot(np.arange(ac.shape[0]), ac[:,1], label = 'long')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('action value')
    plt.title('TimeStamp vs. action')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_path', type=str, default='examples/metadrive/h5py/one_pack.h5py')

    args = parser.parse_args()
    args = vars(args)

    visualize_h5py(args)
