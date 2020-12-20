# DQN Pytorch

This project is a Pytorch implementation of several variants of the Deep Q Learning (DQN) model. It is based on the material provided by Udacity's Deep Reinforcement Learning Nanodegree. The objective is to use one of the Unity ML-Agents libraries to demonstrate how different DQN implementations can be coded, trained and evaluation. 

![Banana Game Screen Shot](./images/trained.gif)

# Sumary

The code structure builds from the Nature DQN, and incrementally implements 3 modifications, in order: Double Q Learning, Duelling Networks and Prioritized Experience Replay. The articles for each one of these implementations can be found at

- DQN [[1]](#references)
- Double DQN [[2]](#references)
- Dueling Network Architecture [[3]](#references)
- Prioritised Experience Replay [[4]](#references)

Although the code can be used in any operating system, the compiled versions of the Unity ML-Agents environment used are only available to MAC (with graphics) and Linux (headless version, for faster training). To download the Linux version with graphics or the Windows versions, please use the links below (provided by Udacitys Nanodeegre):

* Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)



# Dependencies

1. It is recommended to use mini-conda to manage python environments. In order to install the dependencies the initial step would be:

```bash
	conda create --name dqn-pytorch python=3.6
	source activate dqn-pytorch
```

2. The necessary packages to run the code can be obtained by cloning and installing Udacity's Nanodegrees repository (plus, the repo is lots of fun to anyone wanting to explore more projects related to reinforcement learning)
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. To use jupyter notebooks or jupyter lab properly, it is important to create an ipython kernel.
```bash
python -m ipykernel install --user --name dqn-pytorch --display-name "dqn-pytorch"
```

4. Before running code in a notebook, change the kernel to match the `dqn-pytorch` environment by using the drop-down `Kernel` menu. 

# Unity ML-Agents Environment

The environment consists of a robot surround by a boxed enclosure filled with yellow and blue bananas At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state-space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 

# Training and Playing

To get started with the code, the first step is to load the Unity-ML agent's environment. It is important to note that the path must be adjusted to the location of the environment file in your system. The environment is organized around brains that represent each controllable agent. In the banana environment, it suffices to use the first brain. The initial code would be:

```python
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="environments/Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

The next step is to load one of the implemented agents and corresponding training class. For the banana environment, the state size must be 37, and action size 4. The training setup must include the number of episodes, and the values for the epsilon and beta parameters evolution. An example with the values used in the trained models and the Prioritized Replay model is:

```python
from dqn import PriorAgent, PTraining

agent = PriorAgent(state_size=37, action_size=4, seed=0)
training_setup = PTraining(n_episodes=2000, eps_start=1, eps_end=0.01, eps_decay=0.995, beta_start=0.4, beta_inc=1.002)
```

To train the agent and get the scores during training, use the train function of the training class. 

```python
scores = training_setup.train(agent, env, brain_name, track_every=2, plot=True, weights='final.pth',success_thresh=13.)
```

The class receives as inputs:

* the agent 
* the environment, 
* the brain name
* track_every - the number of steps between the tracking of the training
* plot - wether or not the tracking is visual (with an evolution plot) or only informative (with prints)
* success_thresh - The threshold for the moving average of the last 100 runs. When it is conquered, the training stops and the weights are saved in the models folder
* weights - The name of the weights file where the model will be saved


Once the scores is saved, you can save the training with a name and description using the Benchmark class. To do so, just do as the code bellow.

```python
from dqn import  Benchmarks

benchs = Benchmarks()
benchs.save_score('Final Prioritized Replay', scores, 'Prioritized replay implementation, with duelling model and Double DQN, the impletation trained for 2000 episodes'))
```
To check all available saved trainings, check the [Benchmarks](#benchmarks) section. To see a trained model play, just load the weights for the agent with the load_weights function, and use the play function of the training class.

```python
agent = PriorAgent(state_size=37, action_size=4, seed=0)
agent.load_weights('final.pth')
scores = PTraining().play(agent, env, brain_name)
```

Below is a comparison with the Prioritized Replay model, of an untrained agent, with an agent trained for 2000 steps. Check how the trained model is able to search for yellow bananas while avoiding blue ones

![](images/untrained.gif)  |  ![](images/trained.gif)
-------------------------|-------------------------
Untrained Model          |  Trained Model

# Code base

The folder system in the code is structured as:

* benchmarks - Training scores and description of each model already trained

* dqn - Main library, with different implementations of the DQN model

* models - Saved weights of the trained models

* images - Saved images of results

* Navigation.ipynb - Jupyter Notebook with code samples

# DQN library

The DQN library is organized in classes as follows

* Model Modules - Modules to train and use each one of the implementations
* Benchmarks - Class to load and display the saved training scores

Each model module is organized as

* Agent - The agent implementation, responsible to interact and learn with the environment 

* Model - Neural Net implementation in PyTorchh of the DQN architecture

* Training - Convenience  class to handle training and tracking of the agent 

For a description of the implementation of the most complex variant, see the [Report file](./Report.md).

The available models and corresponding classes are:

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
    DQN with modification to implement double q learning and a dueling network architecture
    ```python
        from dqn.duelling import DDDQNAgent, DuelTraining
    ```

* Prioritized Replay \
    DQN with modification to implement double q learning, a dueling network architecture, and Prioritized replay
    ```python
        from dqn.prioritized import PriorAgent, PTraining
    ```

# Benchmarks

The 4 models implemented have trained versions saved in the models folder. Those models are named as:

* Nature DQN [[1]](#references) -> dqn.pth
* Double DQN [[2]](#references) -> ddqn.pth
* Duelling Double DQN [[3]](#references) -> dddqn.pth
* Prioritezed Replay DQN [[4]](#references) -> priordqn.pth
* Prioritezed Replay trained through 2000 steps -> final.pth
* Untraind Prioritized Replay DQN -> untrained.pth


Also, the scores for every training along with a description of the model used are saved in the benchmarks folder. The available scores are:

* DQN -> Nature DQN training
* DDQN -> Double Q learning DQN training
* DDDQN -> Dueling Network with Double Q 
learning DQN training
* Prioritized Replay -> Prioritized Replay (with double q learning and dueling architecture)
* Final Prioritized Replay - Prioritized architecture trained through 2000 step
* random -> Performance of a random agent

To load a specific model, just use the function load_bench from the Benchmarks class. The load class receives the name of the saved scores. To plot the scores, use the plot_bench function. This function receives the scores vector, the title of the plot 

```python
scores = benchs.load('DQN')
bench_dict = benchs.plot_bench(scores, title='Example of Loading Score', mean=100, opacity=0.5)
```

![Example Score Loading](./images/example_score_loading.png)

The plot function receives the scores vector, the title of the plot, the number of runs to use in the moving mean calculation (or None for not displaying the mean) and the opacity to use for the plotting of the scores. 

To see a comparison of all the trainings, you can load a dictionary of { 'model name': [scores vector] } with the load_benchmarks function. To plot the dictionary, use the plot_benchs function

```python
bench_dict = benchs.load_benchmarks()
benchs.plot_benchs(bench_dict, title='Models Comparison', mean=100, opacity=0.1)
```

![Model Comparison](./images/models_comparison.png)

For further details of the implementation of the reinforcement learning agent, the Prioritized Replay model architecture is describe with details in the [Report file](./Report.md).

# References


 
[1] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)

[2] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

[3] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)

[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
 
