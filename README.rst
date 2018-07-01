TorchRL
========
[![Build Status](https://travis-ci.org/activatedgeek/torchrl.svg?branch=master)](https://travis-ci.org/activatedgeek/torchrl)
[![PyPI version](https://badge.fury.io/py/torchrl.svg)](https://pypi.org/project/torchrl/)
![Project Status](https://img.shields.io/badge/status-beta-green.svg)


Objectives
-----------

* Modularity in the RL pipeline

* Clean implementations of basic ideas

* Scalability

Install
--------

```
$ pip install torchrl
```

Usage
------

.. program-output:: torchrl -h

Experiments
------------

## DQN on CartPole-v1

```
$ torchrl --problem=dqn-cartpole-v1 --hparam-set=dqn-cartpole --seed=1 \
    --usr-dirs=experiments --log-dir=log/dqn --show-progress
```


## A2C on CartPole-v0

```
$ torchrl --problem=a2c-cartpole-v0 --hparam-set=a2c-cartpole --seed=1 \
    --usr-dirs=experiments --log-dir=log/a2c --show-progress
```

## DDPG on Pendulum-v0


```
$ torchrl --problem=ddpg-pendulum-v0 --hparam-set=ddpg-pendulum --seed=1 \
    --usr-dirs=experiments --log-dir=log/ddpg --show-progress
```

## PPO on Pendulum-v0

```
$ torchrl --problem=ppo-pendulum-v0 --hparam-set=ppo-pendulum --seed=1 \
    --usr-dirs=experiments --log-dir=log/ppo --show-progress
```

# Resume Experiments

To reload an experiment from previous run, say for instance the DQN run
above for `3000` more steps (optional argumen),

```
$ torchrl --usr-dirs=experiments --load-dir=log/dqn \
    --extra-hparams=num_total_steps=3000 \
    --show-progress
```

This will read all the other parameters from the directory and load the latest
checkpoint.
