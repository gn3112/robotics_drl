# Deep reinforcement learning for simultaneous robotic manipulation and locomotion 

This repo contains a training pipeline for robot learning in a virtual environment for both manipulation and locomotion. With the final goal of deploying this trained agent in the real-world, the [RCAN Sim2Real](https://arxiv.org/abs/1812.07252) pipeline is extended with a novel dataset for joint locomotion and manipulation, which supports transfer of a learned policy from simulation to the real-world (this work can be found [there](https://github.com/gn3112/rcan_navig)).

**Motivation behind this project**: it was realised as part of my MSc thesis with the challenge of being able to control a robot arm and a moving base in an unconstrained environment autonomously. The first step toward this goal were realised with a sample efficient training pipeline combining both manipulation and locomotion motor control.

## Simulation environment
Simulation experiment are performed in VREP using the PyRep toolkit. Two robots were used for experiments in simulation, both with omni-directionnal wheels and >4dof arm on top of the base. The kuka youbot model was provided by VREP and the other custom designed robot model was imported from its CAD model.

## Training algorithm
The soft actor-critic algorithm was chosen as the backbone of our DRL agent. The value-network in the original paper is removed and the value function is computed from the Q-networks.
We sample and fill the replay buffer with demonstration before the start of training. We use prioritize experience replay to sample demonstrations more frequently and sample transitions having a higher temporal difference error with a higher probability.

For the mobile manipulation agent, simply training with a simple fully-connected network on symbolic observations did not perform well. Reasons for this originated from destabil- isation of the mobile platform with simultaneous motion of both the arm and the base and unnecessary exploration for the arm in specific region of the state space. We therefore in- troduced a branch in the network to dampen the effect of one another by simply computing a cut-off value with a sigmoid (Fig. 9).


## Algorithm parameters (hyperparameters)
- Size of the replay buffer: 100k
- 1-step returns
- Auxiliary loss for behavior cloning added to the policy network loss
- Q-filter to account from demonstrator suboptimal actions
- Sparse reward function with a penalizing reward for to high accelerations
- The weight of the behavioural cloning loss is annealed as such: (N DEM/BATCH SIZE) × 0.4
- Also, to reduce the time complexity of the algorithm only the actor is trained on images while the critic is trained on the symbolic observation space, as the critic is not needed during deployment.

## Results

## Relevant papers
-  Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Van- houcke, and Sergey Levine. Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation. CoRR, abs/1806.10293, 2018. URL http://arxiv.org/abs/1806.10293.
- Konstantinos Bousmalis, Alex Irpan, Paul Wohlhart, Yunfei Bai, Matthew Kelcey, Mrinal Kalakrishnan, Laura Downs, Julian Ibarz, Peter Pastor, Kurt Konolige, et al. Using simulation and domain adaptation to improve efficiency of deep robotic grasping. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 4243– 4250. IEEE, 2018
- Joshua Tobin, Rachel H Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 23–30, 2017.
- Stephen James, Marc Freese, and Andrew J. Davison. Pyrep: Bring- ing V-REP to deep robot learning. CoRR, abs/1906.11176, 2019. URL http://arxiv.org/abs/1906.11176.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off- policy maximum entropy deep reinforcement learning with a stochastic actor. CoRR, abs/1801.01290, 2018. URL http://arxiv.org/abs/1801.01290.
- Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, et al. Deep q-learning from demonstrations. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.
- Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Overcoming exploration in reinforcement learning with demonstrations. CoRR, abs/1709.10089, 2017. URL http://arxiv.org/abs/1709.10089.
- Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.
- Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. CoRR, abs/1511.05952, 2015.
- Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954, 2018.
- Matej Veerk, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothrl, Thomas Lampe, and Martin Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse re- wards. 07 2017.
