# Pursuit Evasion

This directory contains pursuit evasion game simulations using the Genesis framework. 

## Training
Currently, one vs one pursuit evasion is supported, where the target follows a potential field strategy, and the pursuer is trained with PPO algorithm provided by `skrl`. The gaming environment can be either a circular arena or an open space, use `arena=true` or `arena=false` in the yaml file to toggle between the two.

Alternatively, specify the environment name in the command line, and the environment will be loaded from the specified yaml file under `configs/`.
```bash
python3 pursuit_train.py [-v] [-e env_name]
```

If logging training data with `wandb`, make a copy of `wandb_config.example.py` and rename to `wandb_config.py`, then set the `wandb` API key in the file.
