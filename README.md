<h1 align="center">基于RecurrentPPO的AKF紧耦合GNSS单点定位增强系统 🛰️</h1>
<h3 align="center">Robust PPO-aided Adaptive Kalman Filter for Single Point Positioning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PYTHON-3.8+-0072B2?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/RL-STABLE%20BASELINES3-800080" alt="RL">
  <img src="https://img.shields.io/badge/CONFIG-HYDRA-FF7F0E" alt="Config">
  <img src="https://img.shields.io/badge/TUNING-OPTUNA-0072B2" alt="Tuning">
</p>

## 📖 项目介绍
本项目为国家自然科学基金（No. 62320106008）资助项目，由北斗 PNT 实验室与企业技术委托合作开发。
项目针对高楼、树荫等城市复杂观测环境下的全球卫星导航系统（GNSS）单点定位（SPP）精度严重衰减、误差激增的问题，提出并实现了一套基于深度强化学习（DRL）的自适应卡尔曼滤波（AKF）增强系统。

系统核心设计采用近端策略优化算法（PPO）以及支持时序处理的 RecurrentPPO (LSTM+PPO)，在动态时序序列中实时学习并调节卡尔曼滤波器的噪声矩阵与卡尔曼增益（Kalman Gain），从而在 NIS（新息平方）域内实现抗差状态估计，大幅提升了复杂路况下的端到端定位精度与鲁棒性。
## ✨ 项目亮点

本项目在设计上兼顾了前沿算法的探索与工业级代码的落地，亮点主要体现在以下两个维度：

### 🧠 算法创新视角 (Algorithmic Innovations)
* **DRL 与传统导航算法的深度耦合：** 摒弃了传统的经验性噪声矩阵调参。系统利用智能 Agent 实时感知动态环境变化，实现对 KF 中卡尔曼增益矩阵的自主、连续调优，从而在 NIS（新息平方）域内实现抗差状态估计。
* **专为时序序列优化的记忆模块：** 针对复杂城市环境下的部分可见马尔可夫决策过程（POMDP），引入带有循环神经网络（LSTM）的 RecurrentPPO 结构，有效捕获 GNSS 卫星观测信号中的历史隐藏状态和长期时序依赖。
* **多维特征融合与多准则奖励驱动：** 观测空间融合了伪距残差、载噪比（C/N0）、高度角等丰富异构特征；同时设计了以 RMSE 衰减为主导、轨迹平滑度为惩罚项的复合奖励函数，引导模型快速且平稳地收敛。

### 🏗️ 工程架构视角 (Engineering Architecture)
* **高度解耦的模块化设计：** 遵循极具扩展性的系统架构范式，将强化学习交互环境（Env）、深度学习模型（Model）与数据清洗管线（Data Pipeline）彻底分离，为二次开发和算法迭代提供极佳的体验。
* **配置驱动的敏捷实验管理：** 接入Hydra框架并全面拥抱 YAML 配置体系，实现超参数设置、数据路径与核心代码的零侵入式分离。结合 Optuna 自动化调参工具，仅需修改配置文件即可轻松管理海量实验和调优任务。
* **自动化ML管道：** 集成 Optuna 引擎，结合 Hyperband/SHA 早停剪枝算法与 SQLite 分布式存储，支持多 GPU 节点下的自动化超参搜索
* **内置全链路 GNSS 底层算法库：** 摆脱对臃肿第三方黑盒的依赖，项目内置了精简且功能完整的底层解算支持（`src/utils/gnss_lib`），涵盖星历文件解析、NMEA/RINEX 格式读取、伪距计算及标准单点定位解算。
## 📁 项目结构
    Plaintext
    DRL4KF/
    ├── config/                  # 核心配置文件夹 (YAML格式)
    │   ├── data_path/           # 数据集路径配置
    │   ├── eval_conf/           # 验证环境配置
    │   ├── learn/               # 训练循环与学习率策略配置
    │   ├── model/               # DRL 模型结构超参数配置
    │   ├── policy/              # 策略网络 (Policy Network) 参数
    │   ├── train_conf/          # 训练环境交互配置
    │   └── tune/                # Optuna 自动化调参配置
    ├── resource/                # 快速启动脚本与辅助资源
    ├── src/                     # 核心源代码目录
    │   ├── datapipeline/        # 数据流管线，负责原始观测值清洗与特征工程
    │   ├── main/                # 程序主入口 (main.py)
    │   ├── model/               # 模型算法层
    │   │   ├── env/             # 自定义 Gym 交互环境 (如 Haige_AKF.py)
    │   │   └── RecurrentPPO/    # 带记忆的 PPO 核心算法实现
    │   └── utils/               # 基础工具类
    │       ├── funcs/           # 通用辅助函数 (IO、日志等)
    │       └── gnss_lib/        # 核心导航底层库 (坐标转换、星历解析、定位解算等)
    └── README.md
## 🚀 快速上手
 1.环境准备
 
建议使用 Conda 创建隔离的虚拟环境（Python 3.8+）：

    conda create -n drl4kf python=3.8
    conda activate drl4kf

2.安装依赖

安装深度学习核心框架及项目依赖

    pip install torch torchvision torchaudio
    pip install -r requirements.txt
3.数据集配置

* 将原始 GNSS 数据（如 GSDC 数据或实测数据）放置在本地目录。

* 打开 `config/data_path/data_selfvel_V2.yaml`，修改其中的数据读取挂载路径。

4.开启训练

进入项目主目录，运行主入口脚本：
`python src/main/main.py`

tips：您可以直接在 `config/learn/learn.yaml` 中修改训练参数，无需改动任何源代码。

5.自动化调参

如需在海量参数空间内寻找最优收敛配置，可根据 `config/tune/tune_conf.yaml` 中的设定，打开tune模式进行相关参数的优化。
## 🎛️ 核心配置系统详解

本项目全面拥抱了 **Hydra** 组合式配置框架。将庞大的系统拆解为多个低耦合的 YAML 文件，实现了超参数设置、数据路径与核心代码的零侵入式分离。

### 1. 全局总控台 (`config.yaml`) 🧠
作为整个配置系统的大脑，它负责统筹所有子模块，并设定最基础的全局变量：
```yaml
defaults:
  - data_path: data_selfvel_V2
  - model: model_conf
  - policy: policy_kwargs
  # ... 其他子模块拼装

random_seed: 1
networkmod: "continuous_lstm"
```
* **模块拼接：** 像搭积木一样无缝拼装数据、环境、模型等模块。想要更换数据集或策略，只需在此处修改一行配置
* **动态日志管理：** 系统会自动根据当前时间戳生成唯一的 TensorBoard 日志文件夹，避免多次训练导致的数据覆盖
### 2.强化学习核心 (model_conf.yaml) ⚙️
配置核心网络架构，网络层数及神经元个数

### 3. 训练配置（train_conf.yaml eval_conf.yaml test_conf.yaml）
配置训练 评估和测试的开启 以及核心参数的配置

## 🧠 核心算法架构
本项目的 DRL Agent 交互过程基于标准的部分可见马尔可夫决策过程（POMDP）构建：

强化学习观测空间 (observation): 融合了多维时空特征，包括当前卫星伪距残差res、载噪比（C/N0）、视距向量LOS vector、高度角AA、方位角度EA 。

动作空间 (Action): 输出连续动作域，用于动态缩放和调整卡尔曼滤波器的卡尔曼增益矩阵。

奖励函数 (Reward): 设计了多准则协同奖励，以解算出的定位坐标与 Ground Truth 之间的 RMSE（均方根误差）衰减量为主导，同时引入轨迹平滑度惩罚项，引导 Agent 寻找最优定位策略。

决策引擎 (Agent): 使用 Recurrent PPO 网络结构，确保在复杂的高动态路况下（如车辆频繁进出隧道、高架桥下）具有优异的泛化能力与快速收敛性。

💡 若对本项目代码或算法逻辑有任何疑问，欢迎提交 Issue 或与开发团队联系。
