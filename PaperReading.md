 

[TOC]

 # Paper Reading



#### **Fast Gradient-Descent Methods for TD Learning with Linear Function Approximation (Sutton, ?)**

- two new algorithms with      
  - better convergence rates: GTD2 & TDC
- compared empirical learning     
  -  rate to that of GTD and conventional TD on one large Computer Go problem      
  - and four smalll problems: 3 random-walk problems and a Boyan-chain problem
- 2 new gradient-based      
  - temporal-difference learning algorithms, proved them convergent with      
  - linear function approximation in a general setting that includes both      
  - on-policy and off-policy learning
- both have time and memory      
  - complexity that is linear in the number of features used in the function      
  - approximation, and faster than GTD
- the first time features like      linear complexity, speed and convergence with off-policy and function      approximation have been achieved in one algorithm.

 

 

 

 

 

#### **Distributed Distributional Deterministic Policy Gradients** **(DeepMind**, 2018)

- Distributed Distributional      Deep Deterministic Policy Gradient algorithm (D4PG)
- combined distributional      perspective on rl adapts its to continuous control
- combined with N-step returns      and prioritized experience replay
- art performance in a wide      range of tasks (simple contro, difficult manipulation tasks, hard      obstacle-based locomotion tasks)
- include the inclusion of a      distributional updates to the DDPG
- combined with the use of      multiple distributed workers all writing into the same replay table
- the use of priority was less      crucial to D4PG on hard problem
- (use of priority definitely      able to increase the performance of D3PG, it can also lead to unstable      updates)

 

 

#### **Learning to Predict by the Methods of Temporal Differences (Sutton, 1988)**

- TD method
- using past experience with an
  - incompletely known system to predict its future behaviour
- assign credit by means of the
  - difference between temporally successive predictions
- proved convergence and 
  - optimality for special cases and related them to supervised-learning
  - methods
- required less memory and less 
  - peak computation
- 2 advantages: more   
  -    incremental therefore easier to compute; tend to make more efficient use      
  - of their experience ( converge faster and produce better predictions )
- faster than      
  - supervised-learning methods
- A general methods for      
  - learning to predict
- a learning method animals may      
  - be using evidenced by some detailed features

 

#### **Human-level control through DRL (DeepMind, nature)**

- DQN
- An powerful agent who can      play classic Atari games
- 2 tricks: experience replay      & iterative update
- ![℃ OBUUOO 一 n  OOUUOO 一 n  8 P > u00  UOunlOAUOO ](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image001.png)
- Single architecture can learn      policies in many different envs and reach human level

 

 

 

#### **A Laplacian Framework for Option Discovery in Reinforcement Learning (Machado)** 

- Proto-value functions
- By introducing eigen purpose,      intrinsic reward functions derived from the learned representations
- Useful for multi tasks <--      discovered      without taking the environment’s rewards into consideration
- Helpful in exploration

 

 

 

#### **Curiosity-driven Exploration for Mapless Navigation with Deep RL (Zhelo, O., Zhang,  J., Tai, L., 2018arxiv)**

- Investigate exploration      strategies of DRL methods to learn navigation policies

- Augment the normal reward for      training with intrinsic reward signals

- Approach is tested in a      mapless navigation setting

- Intrinsic motivation is      crucial for inproving DRL performance

- Proposed method can learn      navigation policies, better generalization capabilities in unknown env

- Architecture: ICM      architecture(The Intrinsic Curiosity Module)

- - ![img](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image002.png)

- A3C

- Reward

- - the first part corresponds       to the inverse model training objective, and the second part to the       forward model prediction error
  - Intrinsic reward
    
  - ![fie prediction error between and through  is then used as the intrinsic reward  (2) ](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image003.png)
  - Extrinsic reward
  - ![aouap pug sȚ  leu*s  •up.1 1euopypeJ1 3!suyłxa augap ](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image004.png)

- Actor-critic

- Details:

- - 2D env

  - An episode terminate after       the agent either reach the goal, or collision or after a maximum of 7000       steps training and 400 testing

  - Hyperparameters: …

  - - Actor-critic        network
                     2 CONV layers(8 filters with stride 2, kernel size 5 and 3        ), ELU 
                     --> 2 FC layers (64 and 16 units), ELU
                     --> 1 LSTM layers of 16 cells
                     action probabilities --> 1 linear layer + softmax
                     value function --> 1 linear layer
    - ICM inverse model
                     3 FC layers (128, 64, 16 units), ELU
                     --> 1 FC layer (32, ELU)
                     --> linear layer + softmax
    - ICM forward model
                     2 layers (64, 32 ), ELU
                     --> linear layer

  - Adam optimizer

  - 3m iter, 15h in CPU

 

 

 

 

 

 

**Learning with Stochastic Guidance for Navigation(DeepMind, 2018)**

- Stochastic switch: allowing      agent to choose between high and low variance policies
             --> train effectively

Balance the learning from exploration or heuristics

- 3 -parts : perception,      controllers(Proportional-intergral-derivative, Obstacle Avoidance, DDPG),      stochastic switch
              

- - ![Switch  Perce on  3X3  3X3  PolicyNet  C Net  PID 0 【 「 ](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image005.png)

- Exploration based on      stochastic switch first, DDPG learns to balance between exploration and      heuristic guidance. Thennavigation is all carried out by DDPG solely. DDPG      learn from the demonstrations given by PID and OA

-  Guidance as positive bias for reducing the      variance of gradient estimators

- Introducing positive bias

- Successfully carry out      navigation task --> robustness and strong generalisation 

- DDPG: an actor-critic      approach in DRL that simultaneously learns the policy and the action-state      value(Q) to assess

 

 

 

**Learning to Drive in a Day (Kendall** ***et all 2018*****)**

- State Space:

- - For simple driving tasks it       is sufficient to use a monocular camera image, together with the observed       vehicle speed and steering angle
  - Treat the image:       a Variational       Autoencoder, using a KL loss and a L2 reconstruction loss

- Reward Space:

- - Forward speed
  - Terminate an episode upon an       infraction of traffic rules

- DDPG

- - Replay buffer
  - Prioritised experience       replay
  - TD error
  - New samples are given       infinite weight to ensure all samples are seen at least once
  - our exploration policy is       formed by adding discrete Ornstein-Uhlenbeck process noise [31] to the       optimal policy

 

- Application of DDPG to real      world driving
- DDPG + VAE ----> Auto      Drive in 20 minutes
- Strongly mean reverting noise      with lower variance is easier to anticipate, whilst higher variance noise      provides better state-action space coverage.
- this reward requires no      further information or maps of the environment
- have shown that a simple      Variational Autoencoder greatly improves the performance of DDPG in the      context of driving a real vehicle

 

 

 

 

**Towards Monocular Vision Based Obstacle Avoidance through Deep Reinforcement Learning (Oxford,  2017)**

- ![img](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image006.png)

- Using Conv residual net to      tansfer RGB image into Depth image stack(but inaccurate in practice, so      added random noise and image blur to get better performance and generalize      ability)

- Using Dueing architecture      based Double DQN to make decision

- - Dueling: two streams of FC       layers to compute the value and advantage functions separately
  - Double: Target network and       Online network

- Actions are defined to      control the linear and angular velocities separately in a discretised      format

- Reward function:   → run as fast as possible
             Penality: Simply rotating on the spot
             Collision: additional punishment of -10![img](file:///C:/Users/吴思源/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image007.png)

 

 

 

#### **Deep Reinforcement Learning with Successor Features for Navigation across Similar Environments**

- Only rely on its onboard      sensors to perform the navigation task without explicit localization,      mapping and path planning procedures

- A successor-feature-based DRL  algorithm

  - To solve the problem that  can't quickly adapt to new situation
  - To make the model transfer  aims to  directly tie the learned       representation between tasks

- Inputs:

  - Both visual and depth
  - CNNs

- Successor feature

  - Reward function can be approximately represented ad a linear combination of learned feature

    $$
    \begin{aligned} Q(\mathbf{s}, \mathbf{a} ; \pi) & \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} \phi\left(\mathbf{s}_{t} ; \theta_{\phi}\right) \cdot \omega | \mathbf{s}_{0}=\mathbf{s}, \mathbf{a}_{0}=\mathbf{a}, \pi\right] \\ &=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} \phi\left(\mathbf{s}_{t} ; \theta_{\phi}\right) | \mathbf{s}_{0}=\mathbf{s}, \mathbf{a}_{0}=\mathbf{a}, \pi\right] \cdot \omega \\ &=\psi^{\pi}(\mathbf{s}, \mathbf{a})^{T} \omega \end{aligned}
    $$

  - the successor representation natually decouples **task specific reward estimation** and the **estimation of the expected occurrece of the features** $\phi(\cdot)$ under the specific policy dynamics

  - lends itself well to transfer learning in secnarios of the 2 kinds of squence of RL problems

    - 1. different environment, same reward function
      2. different environment, different reward function (need to be extended(but with minimal additional memory and computational requirement))
- Simulated experiments

  - 1st, test in 3d environment contains cubic objects and a target for the agent to reach
  - 4 discrete choices: {stand still, turn left (90◦), turn right (90◦), go straight (1m)}
  - agent is a simulated Pioneer-3dx robot moving under a differential drive model (with Gaussian control noise)
  - FOR TRAINING the SF-RL model : SGD with Adam optimizer
  - ![1555226285476](assets/1555226285476.png)
  - train a CNN by SL to directly predict the actions computed by an A* planner from the same visual input that SF-RL receieves
  - train a DQN
  - Experiment:
    - train SF-RL DQN-FixFeature DQN-Finetune CNN
    - transfer to different envrionment
