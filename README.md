# uncertainty
Reinforcement learning with state uncertainty.

To begin, install everything in `requirements.txt`.

### Environments

Environments are set up such that you just need to specify the environment string
as an argument (see `unc.args` for more details).

Changes to this base environment are mostly `gym.Wrapper`s around
this environment, in `unc.envs.wrappers`.
