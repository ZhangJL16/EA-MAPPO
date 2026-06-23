# Hierarchical Actor-Critic HAC PyTorch 中文说明

这个目录是一个紧凑的 PyTorch 版 Hierarchical Actor-Critic，也就是 HAC
实现，基于论文 **Learning Multi-Level Hierarchies with Hindsight**。

这份代码面向目标条件连续控制任务。高层策略在状态空间中提出子目标，
低层策略在固定时间窗口 `H` 内尝试达到这些子目标，最底层策略最终输出
环境可以执行的 primitive action。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

训练默认的 MountainCar HAC agent：

```bash
python train.py
```

测试预训练或默认 checkpoint：

```bash
python test.py
```

这些脚本使用旧版 Gym API：

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

如果你使用较新的 Gym 或 Gymnasium，`reset()` 和 `step()` 可能需要做少量
兼容性修改。

## 仓库结构

```text
.
|-- train.py                         # 训练入口和超参数
|-- test.py                          # 测试入口
|-- HAC.py                           # 层级控制器和 hindsight 逻辑
|-- DDPG.py                          # 每一层使用的目标条件 DDPG
|-- utils.py                         # ReplayBuffer
|-- asset/
|   |-- __init__.py                  # 注册自定义 Gym 环境 ID
|   |-- continuous_mountain_car.py   # 支持目标渲染的自定义 MountainCar
|   |-- pendulum.py                  # 输出原始状态的自定义 Pendulum
|   `-- rendering.py                 # Pyglet 渲染工具
|-- preTrained/                      # actor/critic checkpoint
|-- gif/                             # 演示动画
`-- requirements.txt
```

## 已实现环境

导入 `asset` 会注册两个自定义 Gym 环境：

```python
MountainCarContinuous-h-v1
Pendulum-h-v1
```

`train.py` 和 `test.py` 当前使用：

```python
env_name = "MountainCarContinuous-h-v1"
```

自定义 MountainCar 的状态是：

```text
[position, velocity]
```

自定义 Pendulum 的状态是：

```text
[theta, theta_dot]
```

这和标准 Gym Pendulum 的 observation 不同。标准 Gym Pendulum 通常使用：

```text
[cos(theta), sin(theta), theta_dot]
```

## 主要训练配置

大多数配置都硬编码在 `train.py` 中：

```python
env_name = "MountainCarContinuous-h-v1"
max_episodes = 1000
render = False

k_level = 2
H = 20
lamda = 0.3

gamma = 0.95
n_iter = 100
batch_size = 100
lr = 0.001
```

重要的 HAC 参数：

- `k_level`：层级数量。
- `H`：每一层为了达成目标最多尝试多少步。
- `lamda`：触发低层 subgoal test 的概率。注意代码里变量名写作
  `lamda`，不是 `lambda`。

重要的 DDPG 参数：

- `gamma`：折扣因子。
- `n_iter`：每个 episode 结束后，每一层 DDPG 更新多少次。
- `batch_size`：每次 DDPG 更新从 replay buffer 采样多少 transition。
- `lr`：actor 和 critic 的学习率。

当前 MountainCar 的 bounds 会把 actor 的 `tanh` 输出映射到有效的动作或
状态范围：

```python
primitive action = tanh_output * action_bounds + action_offset
subgoal state    = tanh_output * state_bounds + state_offset
```

对于 MountainCar：

```python
state_bounds = [0.9, 0.07]
state_offset = [-0.3, 0.0]
```

这会把子目标映射到：

```text
position: [-1.2, 0.6]
velocity: [-0.07, 0.07]
```

## 架构

### HAC 封装层

`HAC.py` 会为层级中的每一层构建一个 DDPG 模块：

```python
self.HAC[0] = DDPG(state_dim, action_dim, ...)
self.HAC[1] = DDPG(state_dim, state_dim, ...)
self.HAC[2] = DDPG(state_dim, state_dim, ...)
```

第 `0` 层是 primitive controller：

```text
输入：当前 state + goal
输出：环境 action
```

更高层是 subgoal generator：

```text
输入：当前 state + goal
输出：subgoal state
```

每一层都有自己的 replay buffer：

```python
self.replay_buffer = [ReplayBuffer() for each level]
```

### DDPG actor

`DDPG.py` 里的 actor 是目标条件策略：

```text
输入：concat(state, goal)
MLP： Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh
输出：有界 action 或有界 subgoal
```

网络尺寸：

```text
(state_dim + state_dim) -> 64 -> 64 -> action_dim
```

对高层来说，`action_dim == state_dim`，因为高层的 action 本质上是一个
状态空间中的 subgoal。

### DDPG critic

critic 是 UVFA 风格的目标条件 Q 函数：

```text
输入：concat(state, action, goal)
MLP： Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid
输出：范围在 [-H, 0] 的 Q value
```

有界 Q 值通过下面的方式实现：

```python
return -sigmoid_output * H
```

这对应 HAC 论文里的设定：内部 reward 通常是在目标达成前每步 `-1`，
达成目标时为 `0`，因此 return 可以被限制在 `-H` 到 `0` 之间。

这份实现没有使用 target network。

## HAC Rollout 流程

核心调用是：

```python
agent.run_HAC(env, k_level - 1, state, goal_state, False)
```

它从最高层开始，然后递归调用低层。

对于每一层 `i_level`，`run_HAC()` 最多尝试 `H` 步：

1. 当前层用自己的 DDPG actor 选择一个 action。
2. 如果 `i_level > 0`，这个 action 被解释为 subgoal。
3. 下一层被要求去达成这个 subgoal。
4. 如果 `i_level == 0`，这个 action 会直接传给 `env.step()`。
5. transition 被存入当前层的 replay buffer。
6. 当前尝试序列结束后，再加入 hindsight transition。

对于 3 层模型，递归关系可以理解为：

```text
level 2: 选择服务于最终任务的 subgoal
  level 1: 选择更低层的 subgoal
    level 0: 选择 primitive env action
```

## 探索机制

不处于 subgoal-test 模式时，每一层都会这样探索：

```text
80%：actor 输出 + Gaussian noise
20%：uniform random action/subgoal
```

primitive action 的噪声：

```python
exploration_action_noise = [0.1]
```

subgoal state 的噪声：

```python
exploration_state_noise = [0.02, 0.01]
```

动作和子目标都会被裁剪到配置好的有效范围内。

## Subgoal Testing

`lamda` 控制 subgoal testing：

```python
if np.random.random_sample() < self.lamda:
    is_next_subgoal_test = True
```

在 subgoal test 中，低层必须在不加探索噪声的情况下尝试达成高层提出的
子目标。如果低层没能达到这个被测试的子目标，高层会收到一个惩罚
transition：

```python
(state, action, -H, next_state, goal, 0.0, done)
```

这个机制会训练高层策略，让它不要提出低层无法稳定完成的子目标。

## Hindsight Transition

这份实现主要存储两类 hindsight 数据。

### Hindsight action transition

每次尝试后，当前层会根据 goal 是否达成来存 transition：

```text
reward = 0    如果 goal 达成
reward = -1   如果 goal 未达成
```

对于高层，存入 replay buffer 的 action 会被替换成低层实际到达的状态：

```python
action = next_state
```

这就是 hindsight action 的思想：即使原本提出的 subgoal 没有被精确达成，
系统也会记录低层实际完成了什么。

### Hindsight goal transition

一个尝试序列结束后，这个序列里的所有 transition 会被再复制一份，并把
它们的 goal 替换成最终实际到达的状态：

```python
transition[4] = next_state
```

最后一条复制出来的 transition 会设置为：

```text
reward = 0
gamma = 0
```

这和 HER 的思想类似：agent 也会从自己实际达成过的目标中学习。

## DDPG 更新

每个训练 episode 结束后：

```python
agent.update(n_iter, batch_size)
```

`HAC.update()` 会遍历所有层：

```python
for i in range(self.k_level):
    self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
```

每次 DDPG 更新会采样：

```text
(state, action, reward, next_state, goal, gamma, done)
```

critic target：

```python
next_action = actor(next_state, goal)
target_Q = reward + (1 - done) * gamma * critic(next_state, next_action, goal)
```

critic loss：

```python
MSE(critic(state, action, goal), target_Q)
```

actor loss：

```python
-critic(state, actor(state, goal), goal).mean()
```

所以 actor 的训练目标是选择能让 critic 给出更高目标条件价值估计的
action 或 subgoal。

## Checkpoint

模型保存在：

```text
preTrained/{env_name}/{k_level}level/
```

例如：

```text
preTrained/MountainCarContinuous-h-v1/2level/
```

每一层会保存两个文件：

```text
HAC_MountainCarContinuous-h-v1_level_0_actor.pth
HAC_MountainCarContinuous-h-v1_level_0_crtic.pth
```

文件名里使用的是 `crtic` 而不是 `critic`。这是代码当前的
save/load 约定，不要单独重命名文件，除非同时修改 `DDPG.save()` 和
`DDPG.load()`。

当某个训练 episode 达到最终目标时，`train.py` 还会保存一个带 `_solved`
后缀的 checkpoint。

## 日志

`train.py` 会把 episode reward 写入：

```text
log.txt
```

每一行格式是：

```text
episode,reward
```

控制台也会打印：

```text
Episode: <episode>    Reward: <reward>
```

## 切换任务

如果要切换环境，需要同时修改 `train.py` 和 `test.py`：

1. 修改 `env_name`。
2. 设置正确的最终 `goal_state`。
3. 设置判断目标达成的 `threshold`。
4. 设置 `state_bounds`、`state_offset`、`state_clip_low` 和
   `state_clip_high`。
5. 选择合适的 `k_level` 和 `H`。
6. 确保训练和测试使用的 `directory`、`filename` 一致。

如果要接入新的连续控制任务，环境至少应该提供：

```python
env.observation_space.shape
env.action_space.shape
env.action_space.high
env.reset()
env.step(action)
```

同时，环境 state 本身应该适合作为 goal 向量，因为高层策略会直接在
状态空间里输出 subgoal。

## 已知实现细节

- 超参数没有命令行解析，需要直接编辑 `train.py` 和 `test.py`。
- 代码假设使用旧版 Gym API。
- `random_seed = 0` 会导致 seeding 代码块被跳过，因为代码判断的是
  `if random_seed:`。
- 保存 checkpoint 前目录必须已经存在；否则需要在 `DDPG.save()` 中补上
  `os.makedirs(directory, exist_ok=True)`。
- Replay buffer 在有数据后会有放回地随机采样。
- 渲染依赖 `pyglet`，并且需要可用的 OpenGL/display 环境。

## 参考资料

- 论文：[Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948)
- 原始 TensorFlow HAC 代码：<https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC->
- 这个 PyTorch 实现的来源仓库：<https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch>
