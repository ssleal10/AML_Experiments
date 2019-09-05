# Reinforcement Learning 

## Deep Q-Learning

### Homework
- Play with the hyperparameters and show their corresponding graphs. Which parameter caused the most change? Which one didn’t affect that much? Discuss briefly your results.
- Anneal the <img src="./svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg" align=middle width=6.672392099999992pt height=14.15524440000002pt/> (exploration noise) hyperparameter to decay linearly instead of being fixed. Did it help at all? Why?
- Try two different architectures and report any results

## Deep Deterministic Policy Gradient
DDPG implement DPG algorithm with actor-critic parametrized with neural networks.\\

### Actor ###
Actor estimate policy <img src="./svgs/a0bba743e0d45642c4c3e52b86657915.svg" align=middle width=37.298393549999986pt height=24.65753399999998pt/> which maps from state to action.\\ 

### Critic ###
Critic estimate action value function <img src="./svgs/5b9f673276d4daa369c7ea9c3f51e061.svg" align=middle width=65.84041859999998pt height=24.65753399999998pt/> under policy <img src="./svgs/a0bba743e0d45642c4c3e52b86657915.svg" align=middle width=37.298393549999986pt height=24.65753399999998pt/>.\\

### Homework
- Change DDPG to Mountain car, (May tune a bit the hyperparameters as constant time systems are different, we load both critic-actor weigth so finetune!). Compare with DQN as the environment is the same.\\
  
  - Compare control action from both DQN and DDPG <img src="./svgs/6dbb78540bd76da3f1625782d42d6d16.svg" align=middle width=9.41027339999999pt height=14.15524440000002pt/> in test.\\
  - Compare estabilization time <img src="./svgs/45daa205a2eacb8e053a24d9ae312e8e.svg" align=middle width=12.140467349999989pt height=20.221802699999984pt/> (time to reach goal) in different episodes.\\
  
- **(Optional)** In continous montain car the reward is

<p align="center"><img src="./svgs/3f07748cf5a51330d54ff82a52ca6f11.svg" align=middle width=283.5015645pt height=39.452455349999994pt/></p> 
    - Change the reward so it penalize the velocity of the car.



### Our results (In the pendulum-v0 environment)
![DDPG Performance on swing up pendulum (Pendulum-v0)](./02_DDPG/Trained_Models/DDPG_Pendulum-v0/reward_vs_episode.png)
