# Pursuit Evasion

This directory contains pursuit evasion game simulations using the Genesis framework. 

## Training
Currently, one vs one pursuit evasion is supported, where the target follows a potential field strategy, and the pursuer is trained with PPO algorithm provided by `skrl`. The gaming environment is designed to be a circular arena for now.
```bash
python3 pursuit_train.py [-v] [-B num_envs]
```

## Naive 1v1 Pursuit Evasion
Under `rsl`, 1v1 pursuit evasion is supported, where the target follows a potential field strategy, and the pursuer is trained with PPO algorithm provided by rsl_rl.