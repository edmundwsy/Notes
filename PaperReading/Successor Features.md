# Successor Features

[TOC]



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
  - ![1555226285476](../assets/1555226285476.png)
  - train a CNN by SL to directly predict the actions computed by an A* planner from the same visual input that SF-RL receieves
  - train a DQN
  - Experiment:
    - train SF-RL DQN-FixFeature DQN-Finetune CNN
    - transfer to different envrionment

## 摘要部分

### 研究问题

> Robot navigation in **simple maze-like environments** where only relies on on board sensors

### 出发

#### 传统：SLAM方法

**Drawbacks**

> The majority of SLAM solutions are implemented as *passive procedures* relying on special exploration strategies or a human controlling the robot for sensory data acquisition.(大多数 SLAM 解决方案都是作为被动程序实现的, 依靠特殊的探索策略或人类控制机器人进行感官数据采集。)

> They require an expert to check as to whether the obtained map is accurate enough

#### 本文的目的

> Our goal in this paper is to make first steps towards a solution for navigation tasks without explicit localization, mapping and path planning procedures.

因此作者引入RL

但是：**原来的RL方法都对于迁移到新问题中都太慢了**（估计是需要重新训练）

​	原因？**不满足某个先决条件**（啥？）

所以这个 model 可以自然迁移到序列任务中，并且最小额外损失

> This formulation can be extended to handle sequential task transfer naturally, with minimal additional computational costs

所以需要让模型迁移的更快，使Model能在不同的迷宫中通行，输入不仅仅是visual的，还需要depth的，可以确认CNN的良好效果

> In addition, we validate that deep convolutional neural networks (CNNs) can be used to imitate conventional planners in our considered domain.



## 相关工作

- value-based RL in combination with Deep neural networks 
  Deriving extended variants （生成很多变体）

  > A neural network trained using Q-learning on a specific task is expected to learn features that are informative about both: 
  >
  > 1. the dynamics induced by the policy of the agent in a given environment (we refer to this as the policy dynamics in the following text)
  >
  > 2. the association of rewards to states
  >
  > These two sources of information cannot be assumed to be clearly separated within the network.
  >
  >
  > 更确切地说, 尽管在特定任务上使用 q 学习训练的神经网络有望学习有关这两个方面的信息的功能: 
  >
  > 1. 代理在给定环境中的策略所诱导的动态 (我们将此称为特定环境中的策略动态）
  > 2. reward和states 之间的联系
  >
  > 这两个信息来源不能被认为是在网络内明确分开的。
  >
  > 目前尚不清楚如何以保持原任务policy完整的方式转让上述知识


- 目前尚不清楚如何以保持原任务policy完整的方式转让上述知识
  

### 之前的迁移尝试：

#### 学习一个通用普适的value function

> One attempt at **clearly separating reward** attribution for different tasks while learning a shared representation is the idea of **learning a general (or universal) value function** [14] over many (sub)-tasks that has recently also been combined with DQN-type methods [15].
>
> 
>
> [14] R. S. Sutton, J. Modayil, M. Delp, T. Degris, P. M. Pilarski, A. White, and D. Precup, “Horde: a scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction.” in The 10th International Conference on Autonomous Agents and Multiagent Systems-Volume, 2011.
> [15] T. Schaul, D. Horgan, K. Gregor, and D. Silver, “Universal value function approximators,” in Proc. of the 32nd International Conference on Machine Learning (ICML), 2015.

作者的方法可以被解释成普适value function 的一个特殊形式

#### 微调DQN

> E.g., Parisotto et al. [19] and Rusu et al. [20] performed multitask learning (transferring useful features between different ATARI games) by fine-tuning a DQN network (trained on a single ATARI game) on multiple “related” games.
>
> [19] E. Parisotto, L. J. Ba, and R. Salakhutdinov, “Actor-mimic: Deep multitask and transfer reinforcement learning,” in Proc. of the International Conference on Learning Representations (ICLR), 2016.
> [20] A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” in Proc. of the International Conference on Learning Representations (ICLR), 2016.

#### Progressive Network

> More directly related to our work, Rusu et al. [21] developed the Progressive Networks approach which trains an RL agent to progressively solve a set of tasks, allowing it to re-use the feature representation learned on tasks it has already mastered.
>
> Their derivation has the advantage that **performance on all considered tasks is preserved** but requires an ever growing set of learned representations.
>
> [21] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

#### Successor representation

使Q-Learning 可以被分成两个子任务：

- 学习可靠的能预测奖励的特征
- 估计特征随时间的演变

之前的文献中已经说明在reward改变尺度和含义时，如何加速学习了

本文还包含训练一个 deep auto-encoder





## Background

### RL

写两个方程：

agent 的目标是最大化未来累计预期回报(cumulative expected future reward(with discount factor $\gamma$))


$$
Q(\mathbf{s}, \mathbf{a} ; \pi)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(\mathbf{s}_{t}\right) | \mathbf{s}_{0}=\mathbf{s}, \mathbf{a}_{0}=\mathbf{a}, \pi\right]
$$
因此用期望代替了policy dynamics（这就是value-based的原因？）

在每个transition处，可以使用Bellman equation 去计算Q

使用bellman equation 去选择最优子结构
$$
Q\left(\mathbf{s}_{t}, \mathbf{a}_{t} ; \pi\right)=R\left(\mathbf{s}_{t}\right)+\gamma \mathbb{E}\left[Q\left(\mathbf{s}_{t+1}, \mathbf{a}_{t+1} ; \pi\right)\right]
$$


### SF-RL



