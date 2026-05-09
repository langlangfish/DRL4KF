import logging
from src.utils.package import wraps, time, logging, th, DictConfig, math, ColoredFormatter, os, random, np, initialize, \
    OmegaConf, compose
import gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd

def setup_logging(level = logging.INFO):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 2: WARNING, 3: ERROR
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    h = logging.StreamHandler()  # 输出到 stderr
    fmt = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        # 可选：对 message 二次着色
        secondary_log_colors={
            "message": {
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        },
        style="%"  # 使用 %-style 格式
    )
    h.setFormatter(fmt)
    root.addHandler(h)

    # 业务 logger 只冒泡
    my_logger = logging.getLogger("my_logger")
    my_logger.handlers.clear()
    my_logger.propagate = True



# class VecNormalizeDictObs(VecEnvWrapper):
#     """
#     一个用于 'Dict' 观测空间的包装器，它使用运行均值和标准差
#     来归一化指定的键。
#     这是针对 sb3==1.6.2 的一个变通方法，因为 VecNormalize
#     不支持 Dict 空间的观测归一化。
#     :param venv: 要包装的向量化环境。
#     :param obs_keys: 观测字典中需要被归一化的键 (keys) 的列表。
#     :param clip_obs: (float) 归一化后观测值的最大/最小值。
#     :param epsilon: (float) 避免除零错误的小值。
#     """
#
#     def __init__(self, venv, obs_keys, clip_obs=1.0, epsilon=1e-8):
#         super().__init__(venv)
#
#         # 确认输入空间是 Dict
#         if not isinstance(self.observation_space, gym.spaces.Dict):
#             raise ValueError("VecNormalizeDictObs 只能用于 'Dict' 观测空间。")
#
#         self.obs_keys = obs_keys
#         self.clip_obs = clip_obs
#         self.epsilon = epsilon
#         self.training = True  # 默认处于训练模式
#         self.running_mean_std = {}
#
#         # 为每一个需要归一化的键初始化 RunningMeanStd
#         for key in self.obs_keys:
#             if key not in self.observation_space.spaces:
#                 raise ValueError(f"键 '{key}' 在观测空间中未找到。")
#
#             space = self.observation_space.spaces[key]
#             if not isinstance(space, gym.spaces.Box):
#                 raise ValueError(f"归一化仅支持 Box 空间。键 '{key}' 是 {type(space)}。")
#
#             # 初始化运行统计
#             self.running_mean_std[key] = RunningMeanStd(shape=space.shape)
#
#     def _normalize_obs_dict(self, obs_dict, update_stats):
#         """
#         归一化观测字典中的指定键。
#         :param obs_dict: 从 venv 获取的原始观测字典。
#         :param update_stats: (bool) 是否更新运行统计 (只应在 step 中进行)。
#         """
#         normalized_obs_dict = obs_dict.copy()
#
#         for key in self.obs_keys:
#             obs_data = obs_dict[key]
#
#             # 1. (如果需要) 更新运行统计
#             if self.training and update_stats:
#                 self.running_mean_std[key].update(obs_data)
#
#             # 2. 归一化
#             mean = self.running_mean_std[key].mean
#             var = self.running_mean_std[key].var
#             # 注意：在 1.6.2 中，std 是 np.sqrt(var + epsilon)
#             std = np.sqrt(var + self.epsilon)
#
#             normalized_obs = (obs_data - mean) / std
#
#             # 3. 裁剪 (Clip)
#             normalized_obs = np.clip(normalized_obs, -self.clip_obs, self.clip_obs)
#
#             normalized_obs_dict[key] = normalized_obs
#
#         return normalized_obs_dict
#
#     def step_wait(self):
#         # 1. 获取原始数据
#         obs, rewards, dones, infos = self.venv.step_wait()
#
#         # 2. 归一化观测，并更新统计数据
#         normalized_obs = self._normalize_obs_dict(obs, update_stats=True)
#
#         return normalized_obs, rewards, dones, infos
#
#     def reset(self):
#         # 1. 获取原始重置观测
#         obs = self.venv.reset()
#
#         # 2. 归一化观测，但不更新统计数据
#         normalized_obs = self._normalize_obs_dict(obs, update_stats=False)
#
#         return normalized_obs
#
#     # --- 必须的方法 (用于评估和保存/加载) ---
#
#     def train(self):
#         """将包装器设置为训练模式（更新统计数据）。"""
#         self.training = True
#
#     def eval(self):
#         """将包装器设置为评估模式（不更新统计数据）。"""
#         self.training = False
#
#     def save(self, path: str) -> None:
#         """保存运行均值/标准差到文件。"""
#         save_dict = {
#             key: {
#                 'mean': rms.mean,
#                 'var': rms.var,
#                 'count': rms.count
#             }
#             for key, rms in self.running_mean_std.items()
#         }
#         np.save(path, save_dict)
#         print(f"VecNormalizeDictObs 统计数据已保存到 {path}")
#
#     def load(self, path: str) -> None:
#         """从文件加载运行均值/标准差。"""
#         try:
#             load_dict = np.load(path, allow_pickle=True).item()
#             for key in self.obs_keys:
#                 if key in load_dict:
#                     self.running_mean_std[key].mean = load_dict[key]['mean']
#                     self.running_mean_std[key].var = load_dict[key]['var']
#                     self.running_mean_std[key].count = load_dict[key]['count']
#                 else:
#                     print(f"警告: 在加载的统计数据 '{path}' 中未找到键 '{key}'。")
#             print(f"VecNormalizeDictObs 统计数据已从 {path} 加载")
#         except Exception as e:
#             print(f"加载 VecNormalizeDictObs 统计数据失败: {e}")

def timing_minutes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        minutes = int(duration // 60)
        seconds = duration % 60
        logging.info(f"[⏱] 函数 '{func.__name__}' 执行时间: {minutes} 分 {seconds: .2f} 秒")
        return result

    return wrapper