# Stable Baselines3

This repo is forked from the original stable-baseline3 repo [here](https://github.com/DLR-RM/stable-baselines3).

We adapted this repo to include Safe Reinforcement Learning algorithms, e.g., SAC-lag, and also behavior cloning algorithms.

For example usage, please check out the directory `examples/`. Run the following command in terminal for a demo of SAC.

```shell
python examples/run_bc.py -d examples/ppo_safexp_cargoal1_v0.h5py -e Safexp-CarGoal1-v0 -es 0
```
