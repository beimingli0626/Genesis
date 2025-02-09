# Capture the Flag

This directory contains capture the flag game simulations using the Genesis framework. 

## Training
Currently, one vs one capture the flag is supported, where the target follows a random strategy, and the pursuer is trained with PPO algorithm provided by `skrl`. 

Specify the environment name in the command line, and the environment will be loaded from the specified yaml file under `configs/`.
```bash
python3 pursuit_train.py [-v] [-e env_name]
```

If logging training data with `wandb`, make a copy of `wandb_config.example.py` and rename to `wandb_config.py`, then set the `wandb` API key in the file.
