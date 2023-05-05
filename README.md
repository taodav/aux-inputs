# Agent-State Construction with Auxiliary Inputs
Reinforcement learning with auxiliary inputs.

Code for the corresponding paper published at TMLR.

[Paper](https://openreview.net/forum?id=RLYkyucU6k)

### Installation

Simply install everything in `requirements.txt`.

### Running

All experiments are run through the `main.py` file. Check out the arguments file in `unc/args.py` for a list of all possible hyperparameter configurations.

### Experiments

Experiments are defined in the hyperparameter files located in `scripts/hparams`.

### Environments

Environments are set up such that you just need to specify the environment string
as an argument (see `unc.args` for more details).

Changes to this base environment are mostly `gym.Wrapper`s around
this environment, in `unc.envs.wrappers`.
