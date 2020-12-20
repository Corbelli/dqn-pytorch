# DQN Pytorch

This project is a Pytorch implementation of several variants of the Deep Q Learning (DQN) model. It is based on the material provided by the Udacity's Deep Reinforcement Learning Nanodegree. The objective is to use one of the the Unity ML-Agents library to demonstrate how different DQN implementations can be coded, trained and evaluation. 

The code structure builds from the Nature DQN, and incrementatly implements 3 modifications, in order: Double Q Learning, Duelling Networks and Prioritized Experience Replay.

Altough the code can be used in any operating system, the compiled versions of the Unity ML-Agents environment used are only anvaiable to MAC (with graphics) and Linux (headless version, for faster training). If you are a windows user, please feel free to use the code structure on other compiled environmnents, or use this code in a cloud environmnent on a Linux machine.



## Dependencies

1. I reccomend using mini-conda to manage python environments, so in order to install the dependencies, a
first step would be:

```bash
	conda create --name dqn-pytorch python=3.6
	source activate dqn-pytorch
```

2. The necessary packages to run the code can be obtained by cloning and installing Udacity's Nanodegrees repository (plus, the repo is lots of fun to anyone wanting to explore more projects related to reinformcement learning)
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. To use jupyter notebooks or jupyter lab properly, it is important to create a ipython kernel.
```bash
python -m ipykernel install --user --name dqn-pytorch --display-name "dqn-pytorch"
```

4. Before running code in a notebook, change the kernel to match the `dqn-pytorch` environment by using the drop-down `Kernel` menu. 

## Code base

The folder system in the code is structured as:

* benchmarks - Training scores and description of each model already trained

* dqn - Main library, with different implementations of the DQN model

* models - Saved weights of the trained models

* images - Saved images of results

## DQN library

The DQN libray is organized in classes as follows

* Model Modules - Modules to train and use each one of the implementations
* Benchmarks - Class to load and display the saved training scores

Eacha model module is organized as

* Agent - The agent implementation, responsible to interact and learn with the environment 

* Model - Neural Net implementation in pytorch of the DQN architecture

* Training - Convinience class to handle training and tracking of the agent 

For a description of the implementation of the most complex variant, see the Report document.

The avaiable models and corresponding classes are:

* Nature DQN \
    The original DQN proposed
    ```python
        from dqn.nature import DQNAgent, NatureTraining
    ```

* Double DQN \
    DQN with modification to implement double q learning
    ```python
        from dqn.double import DDQNAgent, DDQNAgent
    ```

* Duelling DQN \
    DQN with modification to implement double q learning and a duelling network architecture
    ```python
        from dqn.duelling import DDDQNAgent, DuelTraining
    ```

* Prioritized Replay \
    DQN with modification to implement double q learning, a duelling network architecture, and Prioritized replay
    ```python
        from dqn.prioritized import PriorAgent, PTraining
    ```

## Unity ML-Agents Environmnet

The environment consists of a robot surround by a boxed enclosure filled with yellow and blue bananas At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 



## Training and Playing

For training and playing structions, please refer to the Tutotial [notebook](./Tutorial.ipynb).
## Results

For futher details of implementation and the results of training, please refer to the Report file [here](./Report.md).