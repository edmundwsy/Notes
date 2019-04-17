# Q-learning

![Q-learning 算法更新](RL_Q-learning.assets/2-1-1.png)

精髓：
$$
Q(s,a) \leftarrow Q(s,a) + \alpha [ r + \max_{a'} Q(s',a') - Q(s,a)]

$$

## 公式推导

Goal:    $ \max E[\sum_{t=0}^ H  \gamma^t R(S_t, A_t,S_{t+1})|\pi]$

> Qlearning的主要优势就是使用了时间差分法TD（融合了蒙特卡洛和动态规划）能够进行离线学习, 使用bellman方程可以对马尔科夫过程求解最优策略