# Reinforcement Learning 

## Deep Q-Learning

### Homework
- Play with the hyperparameters and show their corresponding graphs. Which parameter caused the most change? Which one didn’t affect that much? Discuss briefly your results.
- Anneal the $\epsilon$ (exploration noise) hyperparameter to decay linearly instead of being fixed. Did it help at all? Why?
- Try two different architectures and report any results
\end{itemize}

## Deep Deterministic Policy Gradient
DDPG implement DPG algorithm with actor-critic parametrized with neural networks.\\

### Actor ###
Actor estimate policy $\pi_\theta(s)$ which maps from state to action.\\ 

### Critic ###
Critic estimate action value function $Q_\pi_\theta(s,a)$ under policy $\pi_\theta(s)$.\\

### Homework
- Change DDPG to Mountain car, (May tune a bit the hyperparameters as constant time systems are different, we load both critic-actor weigth so finetune!). Compare with DQN as the environment is the same.\\
  
  - Compare control action $u$ in test in episode with different initial conditions.\\
  - Compare estabilization time $t_s$ (time to reach goal) in different episodes.\\
  
- **(Optional)** In continous montain car the reward is

$$R_t=\left\{\begin{array}{cc}
100 - 0.1||u||_2  \quad & \text{goal reached}\\
-0.1||u||_2 \quad & \text{otherwise} 
\end{array}\right.$$ 
  Change the reward so it penalize the velocity of the car.



### My results
<p align="center">![DDPG Performance on swing up pendulum (Pendulum-v0)](./02_DDPG/Trained_Models/DDPG_Pendulum-v0/reward_vs_episode.png)</p>
