# !/bin/sh

python examples/collect_h5py_data.py --policy_load_dir tensorboard_logs/sac-Safexp-CarButton1-v0_es3_lam0.1/SAC_7/model.pt \
-e Safexp-CarButton1-v0 -es 3 --steps 50000 --output_dir data/sac-Safexp-CarButton1-v0_es3_lam0.1.h5py

python examples/collect_h5py_data.py --policy_load_dir tensorboard_logs/sac-SafetyCarCircle-v0_es3_lam1.0/SAC_11/model.pt \
-e SafetyCarCircle-v0 -es 3 --steps 50000 --output_dir data/sac-SafetyCarCircle-v0_es3_lam1.0.h5py