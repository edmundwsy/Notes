# gym

测试代码

```python
import gym
import time
env = gym.make('CartPole-v0') 
print("Action", env.action_space) 
print("Observation", env.observation_space.high)
print("Observation", env.observation_space.low)
env.reset()  # reset the environment
env.render()	# 重绘环境的一帧。默认模式一般比较友好，如弹出一个窗口。
time.sleep(2)

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
```

|             |      |      |
| ----------- | ---- | ---- |
| CartPole-v0 |      |      |
| Ant-v0      |      |      |
|             |      |      |
| Pendulum-v0 |      |      |



## Notes

1. `env.reset()` will return a vector of observation. 

2. `env.step() `need a parameter of action, sometimes could be an int or a vector; at the same      time, it would return the observation.

3. `env.render() `is used little in the training, for most of the time there is no need to show the picture.

