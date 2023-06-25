# collect pkl env data from waymo raw data

# python examples/metadrive/collect_pkl_from_waymo.py --tfrecord_dir examples/metadrive/tfrecord_20 --pkl_dir examples/metadrive/pkl
python examples/metadrive/waymo_utils.py --tfrecord_dir examples/metadrive/tfrecord_9 --pkl_dir examples/metadrive/pkl_9

# run pkl files in waymo env to collect h5py RL data for offline RL training

python examples/metadrive/collect_h5py_from_pkl.py --pkl_dir examples/metadrive/pkl_9 --h5py_path examples/metadrive/h5py/one_pack_training.h5py

# train offline BC policy 

python examples/metadrive/run_bc_waymo.py --pkl_dir examples/metadrive/pkl_9 --h5py_path examples/metadrive/h5py/one_pack_training.h5py --output_dir examples/metadrive/saved_bc_policy


