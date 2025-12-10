# RLPong

高级强化学习——乒乓球对战大作业

本项目实现了一个基于深度Q网络（DQN）的强化学习智能体，用于玩Atari 2600的Pong游戏。项目使用了NoisyNet进行探索，支持单人对战AI和双人对战模式。

## 环境配置

根据gym-retro仓库的说法，该库最高支持python3.8，因此本项目使用python3.8作为编译器。

主要依赖库可以从仓库根目录下的`requirement.txt`了解。

此外，由于代码`multiplayer.py`中还用到了`FFMpegWriter`，因此还需要安装`FFMpeg`相关的支撑软件（这些软件并不是python的包），可通过如下方式安装：

```bash
conda install -c conda-forge ffmpeg
```

为了能够使用`Pong-Atari2600`这个游戏环境，我们还需要额外安装相应的Roms。安装的具体步骤请参考[issue](https://github.com/openai/retro/issues/60)

### 具体流程

1. 配置Python环境

```bash
conda create --name rlp python=3.8

conda activate rlp
```

2. 安装gym

```bash
pip install gym==0.23.1
```

3. 安装`gym-retro`

```bash
pip install gym-retro==0.8.0
```

4. 安装`ffmpeg`

```bash
conda install -c conda-forge ffmpeg
```

5. 安装`Pong-Atari2600`的Rom

压缩包已给出

解压压缩包到指定文件夹，本人是解压到了项目根目录下的`./Roms`文件夹下

执行命令

```bash
python -m retro.import ./Roms
```

> 配置完毕

## 项目结构

```
pong-dqn/
├── DQN/                    # 核心DQN实现
│   ├── dqn.py            # DQN智能体和网络架构
│   └── noisy_layer.py    # NoisyLinear层（用于探索）
├── utils/                  # 工具模块
│   └── process_obs_tool.py # 观察预处理工具
├── Roms/                   # Atari Pong ROM文件
├── checkpoints/            # 保存的模型检查点
├── videos/                 # 生成的游戏视频
│   └── test/              # 测试视频目录
├── config.py              # 配置文件
├── multiplayer.py         # 主要训练/测试脚本
├── requirement.txt        # Python依赖
└── README.md              # 项目文档
```

## 如何训练

### 1. 单人模式（训练对AI）

训练一个智能体对战游戏内置AI：

```bash
python multiplayer.py -player 1 -memo_1 <模型名称> -total_episode <训练轮数>
```

**示例**：
```bash
python multiplayer.py -player 1 -memo_1 my_pong_agent -total_episode 400
```

**参数说明**：
- `-player 1`: 单人模式
- `-memo_1`: 模型保存的名称（会创建 checkpoints/<memo_1> 目录）
- `-total_episode`: 训练的总轮数（默认400）
- `-horizon`: 帧堆叠的数量（默认4）
- `-skip_frame`: 跳帧数量（默认4）
- `-agent_1`: 使用的智能体类型（默认DQN）

### 2. 双人模式（训练两个智能体对战）

训练两个智能体互相对战：

```bash
python multiplayer.py -player 2 -memo_1 <模型1名称> -memo_2 <模型2名称> -total_episode <训练轮数>
```

**示例**：
```bash
python multiplayer.py -player 2 -memo_1 agent_left -memo_2 agent_right -total_episode 400
```

**注意**：双人模式下，`memo_2`对应的模型必须已经存在（需要先训练好一个模型）。

### 训练过程

- 训练过程中会自动保存表现最好的模型（基于平均奖励）
- 每25轮会进行一次测试，并生成游戏视频
- 训练结束后会在 checkpoints 目录下保存学习曲线图（pong.png）
- 使用经验回放和目标网络稳定训练

## 如何测试

### 1. 测试单人模型

加载训练好的模型进行测试：

```bash
python multiplayer.py -test_mode -player 1 -memo_1 <模型名称> -start_episode_1 <模型轮数>
```

**示例**：
```bash
python multiplayer.py -test_mode -player 1 -memo_1 my_pong_agent -start_episode_1 375
```

### 2. 测试双人模型对战

让两个训练好的模型互相对战：

```bash
python multiplayer.py -test_mode -player 2 -memo_1 <模型1名称> -memo_2 <模型2名称> -start_episode_1 <模型1轮数> -start_episode_2 <模型2轮数>
```

**示例**：
```bash
python multiplayer.py -test_mode -player 2 -memo_1 agent_left -memo_2 agent_right -start_episode_1 350 -start_episode_2 375
```

### 测试输出

- 测试会生成游戏视频，保存在 `videos/<模型名称>/episode_<轮数>.mp4`
- 视频以10 FPS录制，展示智能体的游戏过程
- 控制台会显示游戏信息

## 技术细节

### DQN网络结构
- 3个卷积层用于特征提取
- 2个全连接层（使用NoisyLinear进行探索）
- 输入：84x84的灰度图像，堆叠4帧
- 输出：3个动作的Q值（上、下、发球）

### 观察预处理
- 颜色掩码突出游戏元素
- 帧跳过减少计算量
- 调整为84x84像素
- 转换为灰度图
- 帧堆叠保留时序信息

### NoisyNet探索
- 使用参数化噪声代替ε-greedy
- 在每轮开始前重置噪声
- 无需衰减ε，持续探索

## 依赖包

主要依赖（见 requirement.txt）：
- gym==0.23.1
- gym-retro==0.8.0
- torch==2.4.1
- opencv-python
- matplotlib
- numpy
- tqdm

## 常见问题

1. **ImportError: No module named 'retro'**
   - 确保已正确安装gym-retro并导入ROMs

2. **模型目录已存在错误**
   - 训练时会检查模型目录是否存在，避免覆盖已训练的模型
   - 使用不同的memo名称或删除现有目录

3. **视频生成失败**
   - 确保已安装ffmpeg
   - 检查videos目录是否存在写权限

4. **CUDA相关问题**
   - 代码会自动检测CUDA是否可用
   - 如果没有GPU，会自动使用CPU训练（速度较慢）