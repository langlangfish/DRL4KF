import random

from src.utils.package import SubprocVecEnv,argparse, os, DictConfig, hydra, th, np, Dict, Any, nn, Union, VecNormalize, DummyVecEnv, \
    Path,Monitor,cast,pd,tqdm,datetime,OmegaConf,optuna,deepcopy,math,sys
from src.datapipeline.datapipeline import Data_Pipeline
from src.model.env.Haige_AKF import Kt_Withvel,GPSPosition_continuous_lospos_KG_onlyKtNallcorrect,Kt_Withvel_dt
from src.model.RecurrentPPO.PPO_hao import RecurrentPPO,MyCustomCallback
from src.utils.funcs.lly_utilis import setup_logging
import logging
from src.utils.funcs.haige_utilis import recording_results_ecef_xyz_v5
import pdb

class OptunaCallback(MyCustomCallback):
    """把 MyCustomCallback 产生的指标上报给 Optuna；必要时触发 prune（早停）。"""

    def __init__(self, *args, trial=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self._reported_steps = set()
        self._should_prune = False  # 新增：只记录“应该 prune”的信号
        self._last_report_step = 0  # 新增：上次向 Optuna 报告的环境步数

    # noinspection PyBroadException
    def _on_step(self) -> bool:
        # 先执行原有逻辑（含验证、保存最好模型、日志等）
        cont = super()._on_step()
        if self.trial is not None:
            try:
                step = int(getattr(self.model, 'iteration', 0))  # 改：用env步数
            except Exception as e:
                step = -1
                logging.error(f"{e}")

            # 按“环境步数差值”来节流上报频率（而不是 step % eval_freq）
            if step - self._last_report_step < self.eval_freq:
                return cont
            self._last_report_step = step

            # if step % self.eval_freq != 0:
            #     return cont

            val = self.last_result_dict.get(self.cfg.result_type, np.nan)
            is_finite = np.isfinite(val)
            try:
                if is_finite:
                    float_val = float(val)
                else:
                    float_val = np.nan
                    logging.warning(f"\nval is not finite at step {step}")
            except Exception as e:
                float_val = np.nan
                logging.error(f"{e}")

            # 每个 _reported_steps 汇报一次，避免刷屏
            if step not in self._reported_steps and is_finite:
                self._reported_steps.add(step)
                try:
                    self.trial.report(float_val, step=step)  # 改：以env步数作为 TrialStep
                    if self.trial.should_prune():
                        logging.info(f"\nTrial should be pruned at step {step}")
                        self._should_prune = True
                        return False  # 让 learn() 跳出 while 循

                except Exception as e:
                    logging.error(f"{e}")
        return cont

    def _on_training_end(self) -> None:
        super()._on_training_end()


class General_Trainer:
    def __init__(self, cfg: DictConfig):
        self.switch = None
        self.cfg = cfg
        self.model = None
        self.training_env = None
        self.eval_env = None
        self.args = argparse.Namespace()
        self.data_pipeline = Data_Pipeline(cfg=self.cfg)
        self.env_data = self.data_pipeline.init_env_data(self.cfg.data_path)


    def search_seed(self):
        seed = self.cfg.random_seed
        gpu_id = self.cfg.gpu_id
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

    def process_env(self, venv):

        self.switch = {
            'raw': {
                'norm_obs': False,
                'norm_reward': False,
            },
            'obs_only': {
                'norm_obs': True,
                'norm_reward': False,
            },
            'rwd_only': {
                'norm_obs': False,
                'norm_reward': True,
            },
            'obs_and_rwd': {
                'norm_obs': True,
                'norm_reward': True,
            },
        }[self.cfg.vec_env_mode]
        return VecNormalize(venv=venv, **self.switch)

    def make_env_for_train(self, env_yaml: DictConfig, mode: str, **kwargs):
        def _thunk():
            return self._gen_env(env_yaml, mode, **kwargs)

        return _thunk

    def gen_training_env(self,env_yaml:DictConfig, **kwargs)->Union[DummyVecEnv, VecNormalize]:
        env = [self.make_env_for_train(env_yaml, mode="train", **kwargs) for _ in
               range(self.cfg.model.training_envs)]
        if self.cfg.model.training_envs > 1:
            env = SubprocVecEnv(env, start_method=self.cfg.start_method)
        elif self.cfg.model.training_envs == 1:
            env = DummyVecEnv(env)
        else:
            env = None
        env = self.process_env(env)
        return env

    def gen_eval_env(self,env_yaml:DictConfig,**kwargs)->VecNormalize:
        env = [self.make_env_for_train(env_yaml, mode="eval", **kwargs)]
        env = DummyVecEnv(env)
        logging.info("Wrapping the eval env in a DummyVecEnv.")
        env = self.process_env(env)
        return env



    def initialize_eval_env(self):
        self.eval_env.training = False
        if self.switch.get('norm_obs'):
            self.eval_env.norm_obs = self.switch.get('norm_obs')
            self.eval_env.obs_rms = self.training_env.obs_rms
        if self.switch.get('norm_reward'):
            self.eval_env.ret_rms = self.training_env.ret_rms
            self.eval_env.norm_reward = self.switch.get('norm_reward')
        self.eval_env.seed(self.cfg.model.seed)
        _ = self.eval_env.reset()



    def initialize_train_env(self):
        self.training_env.seed(self.cfg.model.seed)
        # self.training_env.reset()

    def gen_env_dynamic_args(self,mode, **kwargs) -> Dict[str, Any]:
        trajid_list = self.env_data[self.cfg.train_conf.traj_type]
        trajdata_range = [0, int(np.ceil((len(trajid_list) - 1)))]
        dynamic_args = {
            'trajdata_range': trajdata_range,
            'env_data': self.env_data,
            'cfg': self.cfg,
        }
        return dynamic_args

    def _gen_env(self, env_yaml: DictConfig, mode: str , **kwargs):
        self.args.env_dynamic_args = self.gen_env_dynamic_args(mode,**kwargs)
        self.args.env_static_args = env_yaml
        self.args.env_args = dict(
            **self.args.env_dynamic_args,
            **self.args.env_static_args,        )
        env_class = {
            'losposKG': GPSPosition_continuous_lospos_KG_onlyKtNallcorrect,
            'losposKG_vel':Kt_Withvel,
            'losposKG_vel_dt':Kt_Withvel_dt
        }[self.cfg.envmod]
        env = env_class(**self.args.env_args)
        env = Monitor(env)
        return env

    def gen_tensorboard_log_path(self, dir_path: str):
        tensorboard_log = f'{dir_path}/' \
                          f'GPUid={self.cfg.gpu_id}_Mode={self.cfg.vec_env_mode}'
        print(tensorboard_log)
        return tensorboard_log

    def gen_model_dynamic_args(self):
        self.args.log_args = dict(
            dir_path=Path(self.cfg.dir_path) / self.cfg.tensorboard_log_path,
        )
        logs_dir = self.gen_tensorboard_log_path(**self.args.log_args)
        dynamic_args = {
            'env': self.training_env,
            'tensorboard_log': logs_dir,
            'learning_rate': self.cfg.haige_learning_rate,
            'policy_kwargs': {
                'net_arch': [dict(pi=[self.cfg.policy.pi_network_unit for _ in range(self.cfg.policy.pi_laynum)],
                                  vf=[self.cfg.policy.vf_network_unit for _ in range(self.cfg.policy.vf_laynum)])],
                'activation_fn': nn.SiLU,
            }
        }
        return dynamic_args

    def gen_model(self):
        self.args.model_dynamic_args = self.gen_model_dynamic_args()
        self.args.model_static_args = self.cfg.model
        self.args.model_args = dict(
            # model_data,
            **self.args.model_dynamic_args,
            **self.args.model_static_args,
        )
        model = RecurrentPPO(**self.args.model_args)
        self.model = model
        return model
    def learn_optuna(self, trial=None):
        log_path = Path(self.cfg.dir_path) / self.cfg.log_path
        self.model.eval_env = self.eval_env
        if trial is None:
            callback = MyCustomCallback(model=self.model, eval_env=self.eval_env, training_env=self.training_env,
                                        cfg=self.cfg)
        else:
            callback = OptunaCallback(
                model=self.model, eval_env=self.eval_env, cfg=self.cfg,training_env=self.training_env,
                 trial=trial)
        self.callback = callback
        self.model.learn(log_path=log_path, callback=callback, eval_env=self.eval_env, **self.cfg.learn)
        # 返回回调里记录的“本轮最好指标”，让 objective() 能拿到
        metric = getattr(callback, 'best_mean_err_reduction', None)
        should_prune = getattr(callback, '_should_prune', False)
        return metric, should_prune
    def learn(self,):
        log_path = Path(self.cfg.dir_path) / self.cfg.log_path
        self.model.aux_coef = self.cfg.model.aux_coef
        self.model.eval_env = self.eval_env
        callback = MyCustomCallback(model=self.model, eval_env=self.eval_env,training_env = self.training_env , cfg=self.cfg)
        self.model.learn(log_path=log_path, callback=callback, eval_env=self.eval_env, **self.cfg.learn)
        return self.model.save(Path(self.cfg.dir_path) /self.cfg.best_model_save_path/f'final_model')

class General_Tester(General_Trainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.tmp_traj_range = None
        self.test_trajID_list = None
        self.test_traj_range = None
        self.test_traj = None
        self.trajdata_range = None
        self.cfg = cfg
        self.switch = None
        self.model = None
        self.training_env = None
        self.eval_env = None
        self.args = argparse.Namespace()
        self.test_log_path = Path(self.cfg.dir_path) / 'logs' / Path(
            self.cfg.test.model_pth.split('/')[-3]) / 'test' / Path(self.cfg.test.model_pth.split('/')[-1])
        # if not self.cfg.is_multi_gpu:
        #     self.cfg.log_path = f"logs/test/{self.cfg.test.model_pth.split('/')[-1]}"
        #     self.cfg.test_result_path = f"logs/test"
        #     self.cfg.test_tensorboard_path = f"logs/test/tensorboard"
        self.callback = None
        self.tripID_range = None
        self.test_trip = 0
        self.test_env = None
        self.data_pipeline = Data_Pipeline(cfg)
        self.env_data = self.data_pipeline.init_env_data_traj_all(data=self.cfg.data_path, mode="test")
        self.args = argparse.Namespace()
        # self.model_raw_pth = Path(cfg.dir_path) / Path(cfg.best_model_save_path)
        self.verbose = cfg.model.verbose
        # 设置样式
        self.tqdm_config = {
            'desc': "总任务",
            'position': 0,
            'unit': "iter",
            'colour': "cyan",  # 将进度条颜色设置为青色
            'bar_format': "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        }

    def gen_env_dynamic_args(self, **kwargs) -> Dict[str, Any]:
        self.tmp_traj_range = [self.test_trip, self.test_trip]  # 某场景下，一条一条地测试轨迹
        dynamic_args = {
            'trajdata_range': self.tmp_traj_range,
            'traj_type': kwargs['traj_type'] if 'traj_type' in kwargs.keys() else 'traj_full',
            'env_data': self.env_data,
            'cfg': self.cfg,
        }
        return dynamic_args

    def gen_model_dynamic_args(self, **kwargs):
        dynamic_args = {
            'env': self.test_env,
            'policy_kwargs': {
                'net_arch': [dict(pi=[self.cfg.policy.pi_network_unit for _ in range(self.cfg.policy.pi_laynum)],
                                  vf=[self.cfg.policy.vf_network_unit for _ in range(self.cfg.policy.vf_laynum)])],
                'activation_fn': nn.SiLU,
            },
            'cfg': self.cfg,
        }
        return dynamic_args

    def gen_model(self, model_pth: Path = None) -> RecurrentPPO:
        self.args.model_dynamic_args = self.gen_model_dynamic_args()
        self.args.model_static_args = self.cfg.model
        self.args.model_args = dict(
            # model_data,
            **self.args.model_dynamic_args,
            **self.args.model_static_args,
        )
        # model_type = dict() TODO : 扩展更多模型类型
        model = RecurrentPPO(**self.args.model_args)
        if not model_pth.is_file():
            model_pth = self.model_raw_pth / self.data_pipeline.find_pth("model_pth")
        model.load(path=model_pth, env=self.test_env)
        logging.info(f"Load model from {model_pth}")
        return model

    def gen_test_env(self, env_yaml: DictConfig,**kwargs) -> Union[DummyVecEnv, VecNormalize]:
        """
        参数传入静态参数与数据，函数内生成动态参数，返回结果
        :return:
        """
        self.args.env_dynamic_args = self.gen_env_dynamic_args(**kwargs)
        self.args.env_static_args = env_yaml
        self.args.env_args = dict(
            **self.args.env_dynamic_args,
            **self.args.env_static_args,
        )
        env_class = {
            'losposKG': GPSPosition_continuous_lospos_KG_onlyKtNallcorrect,
            'losposKG_vel':Kt_Withvel,
            'losposKG_vel_dt': Kt_Withvel_dt
            # 'losposcovR_onlyRNallcorrect_v2': GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2,
        }[self.cfg.envmod]
        env = env_class(**self.args.env_args)
        env = Monitor(env)
        logging.info("Wrapping the env with a `Monitor` wrapper")
        env = DummyVecEnv([lambda: env])
        logging.info("Wrapping the env in a DummyVecEnv.")
        self.switch = {
            'raw': {
                'norm_obs': False,
                'norm_reward': False,
            },
            'hand_make': {
                'norm_obs': False,
                'norm_reward': False,
            },
            'hand_make_and_rwd': {
                'norm_obs': False,
                'norm_reward': True,
            },
            'obs_only': {
                'norm_obs': True,
                'norm_reward': False,
            },
            'rwd_only': {
                'norm_obs': False,
                'norm_reward': True,
            },
            'obs_and_rwd': {
                'norm_obs': True,
                'norm_reward': True,
            },
        }[self.cfg.vec_env_mode]
        env = VecNormalize(env, training=False, **self.switch)
        # env = self.process_env(env)
        return env

    # def process_env(self, venv) -> Union[DummyVecEnv, VecNormalize]:
    #     env_pth = Path(self.cfg.test.env_pth)
    #     if not env_pth.is_file():
    #         env_pth = self.model_raw_pth / self.data_pipeline.find_pth("env_pth")
    #     venv = VecNormalize.load(env_pth.as_posix(), venv)
    #     venv.training = False
    #     unwrap_env = cast(GPSPosition_continuous_lospos_KG_onlyKtNallcorrect, venv.venv.envs[0].env)
    #     unwrap_env.__init__(**self.args.env_args)
    #     return venv

    # def gen_callback(self) -> MyCustomCallback:
    #     return MyCustomCallback(model=self.model, eval_env=self.test_env, cfg=self.cfg, cfg_mode=self.cfg.test,
    #                             verbose=self.cfg.verbose, args=self.args)

    def record_results(self, results: Dict[str, Any]) -> None:
        df = pd.DataFrame(results)
        df['3D_Error_Reduction_%'] = ((df['or_3D_err'] - df['rl_3D_err']) / (df['or_3D_err'] - 1e-8)) * 100
        df['ll_Error_Reduction_%'] = ((df['or_ll_err'] - df['rl_ll_err']) / (df['or_ll_err'] - 1e-8)) * 100
        df['3D_err_RMSE_Reduction_%'] = ((df['or_3D_err_RMSE'] - df['rl_3D_err_RMSE']) / (df['or_3D_err_RMSE'] - 1e-8)) * 100
        df['2D_err_RMSE_Reduction_%'] = ((df['or_ll_err_RMSE'] - df['rl_ll_err_RMSE']) / (df['or_ll_err_RMSE'] - 1e-8)) * 100
        mean_series = df.mean(numeric_only=True)
        avg_df = pd.DataFrame(mean_series, columns=['avg']).T
        avg_df['Type'] = 'average'
        df_with_avg = pd.concat([df, avg_df], ignore_index=True)
        file_name = f"VecMode={self.cfg.vec_env_mode}_2D_err={avg_df['2D_err_RMSE_Reduction_%'].iloc[0]: .4}%.csv"
        file_pth = Path(self.cfg.dir_path) /'logs'/Path(self.cfg.test.model_pth.split('/')[-3])/'test'/Path(self.cfg.test.model_pth.split('/')[-1])
        os.makedirs(file_pth, exist_ok=True)
        df_with_avg.to_csv(file_pth / file_name, index=False)
        logging.info(f'Test Finish! Results are saved in {file_pth}')

    def on_test_end(self) -> None:
        self.callback.on_test_end()


    def reinit_test_env(self):
        """
        注意自定义env在venv.venv.envs[0]里
        :return:
        """
        env = self.test_env.venv.envs[0].unwrapped

        self.trajdata_range = [self.test_traj, self.test_traj]  # 某场景下，一条一条地测试轨迹
        env.trajdata_range = self.trajdata_range
        env.traj_idx = self.trajdata_range[0]


    def collecting_features(self, test_type:str) -> None:
        logging.info(f'Starting more test {test_type}...')
        self.test_trajID_list = self.env_data[test_type]
        self.test_traj_range = range(0, len(self.test_trajID_list))
        self.test_env = self.gen_test_env(env_yaml=self.cfg.test.test_indomain,traj_type=test_type)
        self.model = self.gen_model(Path(self.cfg.test.model_pth))
        pbar = tqdm(self.test_traj_range, desc="测试进度", ncols=100, ascii=True)

        for test_traj in pbar:
            self.test_traj = test_traj
            self.reinit_test_env()
            obs = self.test_env.reset()
            max_iter = 10000
            _states = None

            for _iter in range(max_iter):
                action, _states = self.model.predict(obs, deterministic=True, state=_states)
                obs, rewards, done, info = self.test_env.step(action)

                if done:
                    # 更新进度条描述，而不是直接打印
                    pbar.set_description(f"轨迹{test_traj}: 奖励{rewards} 迭代{_iter}")
                    # 如果需要详细日志，使用 write
                    tqdm.write(f"详细信息 - 轨迹{test_traj}: 奖励{rewards}, 迭代{_iter}")
                    break
            else:
                # 如果循环正常结束（未break）
                pbar.set_description(f"轨迹{test_traj}: 未完成, 奖励{rewards}")
                tqdm.write(f"警告 - 轨迹{test_traj} 达到最大迭代次数 {max_iter}")

        pbar.close()
        # for test_traj in tqdm(self.test_traj_range):
        #     self.test_traj = test_traj
        #     self.reinit_test_env()
        #     obs = self.test_env.reset()
        #     max_iter = 10000
        #     _states = None
        #     for _iter in range(max_iter):
        #         action, _states = self.model.predict(obs, deterministic=True, state=_states)
        #         obs, rewards, done, info = self.test_env.step(action)
        #         if done:
        #             print(f"Iter {_iter}, traj {info[0]['traj_idx']} reward is {rewards}, done")
        #             break

    def test(self, test_type: str, results: Dict[str, Any]) -> None:
        result_dict = recording_results_ecef_xyz_v5(
            test_type=test_type,
            data_truth_dic=self.env_data['data_truth_dic'],
            eval_tripID_loop_range=self.test_traj_range,
            tripIDlist=self.test_trajID_list,
            # logdirname=Path(self.cfg.dir_path) / Path(self.cfg.log_path),
            logdirname=self.test_log_path,
            baseline_mod=self.cfg.test.test_indomain.baseline_mod,
            _traj_record=self.cfg.test.traj_record,
            verbose=self.cfg.model.verbose
        )
        for key, value in result_dict.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
        logging.info(f'More Test {test_type} finished.')

class General_Tuner(General_Trainer):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_pruner(self):
        cfg = self.cfg
        name = str(cfg.tune.get('pruner', 'median')).lower()

        if name in ('hyperband', 'hb'):
            # 允许 'auto' 或 int；传给 Optuna 原样即可
            min_resource = cfg.tune.get('hb_min_resource', 'auto') // 5
            max_resource = cfg.tune.get('hb_max_resource', 'auto')
            reduction_factor = int(cfg.tune.get('hb_reduction_factor', 3))
            bootstrap_count = int(cfg.tune.get('hb_bootstrap_count', 0))

            return optuna.pruners.HyperbandPruner(
                min_resource=min_resource,  # 'auto' 或 int(>=1)
                max_resource=max_resource,  # 'auto' 或 int(>=min_resource)
                reduction_factor=reduction_factor,  # 常见 3 或 4
                bootstrap_count=bootstrap_count,  # 0 即可
            )

        elif name in ('sha', 'successive_halving'):
            min_resource = cfg.tune.get('sha_min_resource', 'auto')
            reduction_factor = int(cfg.tune.get('sha_reduction_factor', 3))
            min_es_rate = int(cfg.tune.get('sha_min_early_stopping_rate', 0))

            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource=min_resource,  # 'auto' 或 int
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_es_rate,
            )

        elif name in ('median', 'median_pruner'):
            return optuna.pruners.MedianPruner(
                n_startup_trials=int(cfg.tune.get('n_startup_trials', 5)),
                n_warmup_steps=int(cfg.tune.get('n_warmup_steps', 10_000) * 0.1),
                interval_steps=int(cfg.tune.get('interval_steps', 5_000) * 0.05),
            )

        elif name in ('none', 'nop', 'disabled'):
            return optuna.pruners.NopPruner()

        else:
            raise ValueError(f"Unknown pruner: {name}")

    def optuna_study(self):
        cfg = self.cfg
        # excel_rows = []
        # sampler & pruner中断器

        # 为不同 GPU 进程设置不同采样器种子；并开启并行防重复
        # sampler_seed = int(cfg.tune.get('seed_base', 42)) + int(cfg.gpu_id)
        sampler_seed = int(cfg.tune.get('seed_base', 42))
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            n_startup_trials=int(cfg.tune.get('n_startup_trials', 10)),
            multivariate=True,  # 可选：更强地联合采样（适合多参）
            constant_liar=True  # 关键：并行时减少重复参数
        )

        pruner = self.build_pruner()
        storage_url = cfg.tune.get('storage', None)
        # 2. 【新增】核心逻辑：检查并自动创建父目录
        if storage_url and storage_url.startswith('sqlite'):

            db_file_path = storage_url.replace('sqlite:///', '')

            # 获取文件夹路径 (去掉文件名)
            db_dir = os.path.dirname(db_file_path)

            # 如果路径存在且不为空，则创建文件夹
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
                # print(f"[Info] 已自动创建/确认数据库目录: {db_dir}")

        # 3. 再初始化 Storage
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60.0}}  # 写锁等待
        )
        study = optuna.create_study(
            study_name=cfg.tune.get('study_name', 'ppo_recurrent_tune'),
            direction=cfg.tune.get('direction', 'maximize'),
            sampler=sampler, pruner=pruner,
            storage=storage,
            load_if_exists=cfg.tune.get('load_if_exists', True),
        )

        # noinspection PyBroadException
        def objective(trial: "optuna.trial.Trial"):
            # t0 = time.time()
            # 每个 trial 独立复制一份 cfg，避免污染
            cfg_trial = deepcopy(cfg)

            # 关闭 schedule（否则建议值会被 schedule 覆盖）
            if cfg_trial.get('tune') and cfg_trial.tune.get('overwrite_schedule', True):
                try:
                    cfg_trial.schedule = False
                except Exception:
                    logging.warning("cfg_trial.schedule=False not set")

            # 唯一试验名、seed 与时长
            try:
                cfg_trial.ex = f"Optuna_tune_{trial.number}"
            except Exception:
                logging.warning("cfg_trial.ex not found")

            try:
                base_seed = int(cfg_trial.model.seed)
            except Exception:
                base_seed = cfg_trial.tune.get('seed_base', 42)
            try:
                # cfg_trial.model.seed = int(base_seed) + int(trial.number)
                cfg_trial.model.seed = int(base_seed)
            except Exception:
                logging.warning("cfg_trial.model.seed not set")

            # 你的 PPO 中，这里是“迭代次数”而非 env-steps
            try:
                cfg_trial.learn.total_timesteps = cfg_trial.tune.get('trial_timesteps', 300)
                if hasattr(cfg_trial.learn, 'eval_freq'):
                    # 特别注意，eval_freq小于num_timesteps时确保num_timesteps % eval_freq=0
                    # eval_freq大于num_timesteps时确保num_timesteps % eval_freq ！=0
                    cfg_trial.learn.eval_freq = max(1, int(cfg_trial.learn.total_timesteps // 4))
            except Exception:
                logging.warning("cfg_trial.learn.total_timesteps not set")

            # =============== 试验空间（可删改/扩展）================
            sfloat = lambda n, lo, hi, log=False: trial.suggest_float(n, lo, hi, log=log)
            sint = lambda n, lo, hi, step=1: trial.suggest_int(n, lo, hi, step=step)
            scat = lambda n, ch: trial.suggest_categorical(n, ch)

            # ==== 更新跨度选项 ====
            try:
                cfg_trial.tune.params.learning_rate = sfloat('learning_rate', 1e-5, 3e-3, log=True)
                cfg_trial.model.n_epochs = sint('n_epochs', 5, 20)
                cfg_trial.tune.params.clip_range = sfloat('clip_range', 0.2, 1.)
                cfg_trial.tune.params.clip_range_vf = sfloat('clip_range_vf', 0.2, 1.)
                cfg_trial.model.target_kl = sfloat('target_kl', 0.01, 0.13)
                current_traj_len = sint('traj_len', 5, 15)
                # 2. 显式打印一下，确认这个值是多少
                logging.info(f"DEBUG: Selected traj_len = {current_traj_len}")

                # 3. 统一赋值
                cfg_trial.train_conf.traj_len = current_traj_len
                cfg_trial.eval_conf.traj_len = current_traj_len
                # cfg_trial.train_conf.traj_len = sint('traj_len', 5, 15)
                # # cfg_trial.eval_conf.traj_len = sint('traj_len', 5, 15)
                # cfg_trial.test.test_indomain.traj_len = sint('traj_len', 5, 15)
            except Exception as e:
                logging.warning(f"tune.py: 缺少参数 {e}")

            # ==== 稳定性选项 =============
            # try:
            #     # cfg_trial.train.training_envs = scat('training_env', [1, 4, 8])
            #     # cfg_trial.model.n_steps = scat('n_steps', [128, 256, 512, 1024])
            #     # cfg_trial.model.batch_size = scat('batch_size', [128, 256, 512])  # 注意 batch<=n_steps*n_envs
            #     # cfg_trial.model.gamma = sfloat('gamma', 0.98, 0.999)
            #     # cfg_trial.model.gae_lambda = sfloat('gae_lambda', 0.95, 0.98)
            #     # cfg_trial.tune.params.vf_coef = sfloat('vf_coef', 400., 2000.)
            #     # cfg_trial.model.max_grad_norm = sfloat('max_grad_norm', 0.3, 1.0)
            # except Exception as e:
            #     logging.warning(f"tune.py: 缺少参数 {e}")

            # ==== 探索选项 =============
            try:
                cfg_trial.tune.params.ent_coef = sfloat('ent_coef', 0.0, 0.1)
                # cfg_trial.model.policy.use_sde = scat('use_sde', [True, False])
                # cfg_trial.model.sde_sample_freq = scat('sde_sample_freq', [1, 4, 8])
                # cfg_trial.model.policy.log_std_init = sfloat('log_std_init', -2.0, 2.0)
                # cfg_trial.model.policy.squash_output = scat('squash_output', [True, False])
                cfg_trial.train_conf.continuous_action_scale = sfloat('continuous_action_scale', 1.0, 10.0)
                # cfg_trial.train_conf.noise_scale_dic['measurement'] = sfloat('noise_measurement', 1e-2, 1e-1)
                # cfg_trial.env.eval_env.noise_scale_dic['measurement'] = sfloat('noise_measurement', 1e-2, 1e-1)
                # cfg_trial.eval_conf.continuous_action_scale = sfloat('continuous_action_scale', 1.0, 10.0)
                cfg_trial.train_conf.kg_action_scale['diagonal'] = sfloat('kg_diagonal', 1e-5,1e-3)
                cfg_trial.train_conf.kg_action_scale['down'] = sfloat('kg_down', 1e-7, 1e-5)
                cfg_trial.train_conf.kg_action_scale['up'] = sfloat('kg_up', 1e-10, 1e-8)

            except Exception as e:
                logging.warning(f"tune.py: 缺少参数 {e}")

            # policy 架构
            # try:
            #     cfg_trial.model.policy.pi_network_unit = scat('pi_units', [256, 512, 1024])
            #     cfg_trial.model.policy.pi_laynum = scat('pi_layers', [2, 4, 6])
            #     cfg_trial.model.policy.vf_network_unit = scat('vf_units', [256, 512, 1024])
            #     cfg_trial.model.policy.vf_laynum = scat('vf_layers', [2, 4, 6])
            # except Exception as e:
            #     logging.warning(f"tune.py: 缺少参数 {e}")
            logging.info(f"\n======================= tune.py: 搜索参数 {trial.number} ==============================")
            logging.info(f"可见的GPU: {cfg_trial.gpu_id}")
            logging.info(OmegaConf.to_yaml(trial.params, resolve=True))
            logging.info(f"\n======================= tune.py: 搜索参数 {trial.number} ==============================")
            # ===== 构建并运行单个 trial =====
            cfg_trial.tune.trial_num = trial.number
            trainer = General_Trainer(cfg=cfg_trial)
            trainer.training_env = trainer.gen_training_env(env_yaml=cfg_trial.train_conf)
            trainer.initialize_train_env()
            trainer.eval_env = trainer.gen_eval_env(env_yaml=cfg_trial.eval_conf)
            trainer.initialize_eval_env()
            trainer.model = trainer.gen_model()
            metric = None
            should_prune = None
            try:
                metric, should_prune = trainer.learn_optuna(trial=trial)
            except Exception as e:
                metric = None
                should_prune = None
                logging.error(f"tune.py: 训练过程异常 {e}")
                import traceback
                traceback.print_exc()

                # 2. 【关键】如果在终端直接运行，这里会暂停，让您进入调试模式
                # 注意：这行代码只有在您不使用 nohup 时才有效
                print("检测到异常，进入 PDB 调试模式...")
                pdb.post_mortem(sys.exc_info()[2])

                # 3. 调试完后，继续抛出异常，结束程序
                raise e
            finally:
                # —— 双保险清理（即便出现异常）——
                try:
                    trainer.callback.on_training_end()
                except Exception as e:
                    logging.warning(f"tune.py: 回调清理失败 {e}")

                # 2) 关闭环境（训/评），兼容 DummyVecEnv / SubprocVecEnv / VecNormalize
                for env_name in ('training_env', 'eval_env'):
                    env = getattr(trainer, env_name, None)
                    if env is not None:
                        try:
                            if hasattr(env, 'close'):
                                env.close()
                        except Exception as e:
                            logging.warning(f"cleanup: {env_name}.close failed: {e}")
                        try:
                            # 释放 VecNormalize 包装的底层引用（若有）
                            if hasattr(env, 'venv') and hasattr(env.venv, 'close'):
                                env.venv.close()
                        except Exception as e:
                            logging.warning(f"cleanup: {env_name}.venv.close failed: {e}")

                try:
                    if hasattr(trainer.model.writer, 'close'):
                        trainer.model.writer.close()
                except Exception as e:
                    logging.warning(f"tune.py: 日志关闭失败 {e}")

                # 4) 断开强引用，利于 GC
                try:
                    trainer.model = None
                    trainer.training_env = None
                    trainer.eval_env = None
                    trainer.callback = None
                    del trainer
                except Exception:
                    logging.warning(f"tune.py: 断开强引用失败")

                import gc
                gc.collect()

                try:
                    if th.cuda.is_available():
                        th.cuda.empty_cache()
                        th.cuda.synchronize()
                except Exception as e:
                    logging.warning(f"cleanup: cuda empty_cache/sync failed: {e}")

            if should_prune:
                raise optuna.exceptions.TrialPruned()

            # 兜底校验
            if metric is None or not (isinstance(metric, (int, float)) and math.isfinite(metric)):
                raise optuna.exceptions.TrialPruned("invalid metric")

            # trial_row = {
            #     'trial_number': trial.number,
            #     **trial.params,
            #     'Score_Mean': metric,
            #     'trial_state': getattr(trial, 'state', None).name if getattr(trial, 'state', None) else 'FINISHED',
            #     'duration_s': round(time.time() - t0, 3),
            # }
            # excel_rows.append(trial_row)

            return float(metric)

        study.optimize(
            objective,
            n_trials=cfg.tune.get('n_trials', 20),
            timeout=cfg.tune.get('timeout', None),
            gc_after_trial=True,
            show_progress_bar=cfg.tune.get('progress_bar', True),
            n_jobs=cfg.tune.get('n_jobs', 1),
        )
        # result_pth = cfg.tune.result_pth
        # # Task = "KF"
        # Task = cfg.tune.task
        # df_results = pd.DataFrame(excel_rows)
        # df_results.to_excel(f"{result_pth}/{Task}_{cfg.ex}_results_{cfg.model.seed}.xlsx", index=False)
        return study

class General_Reader(General_Trainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def read_db(self,
                db_path: str = "sqlite:////mnt/sdb/home/dingweizu/langlangfish/long/tune/v1_baseline/RPPO_trial_${.trial_timesteps}.db",
                study_name: str = "ppo_recurrent_tuning",
                read_lines_num: int = 10,
                ):
        storage = optuna.storages.RDBStorage(url=db_path)
        study = optuna.load_study(study_name=study_name,
                                  storage=storage)
        df = study.trials_dataframe(attrs=("number", "state", "value", "params", "duration"))

        # ==== 直接从 Optuna 数据库拉取并保存 Top-N ====
        try:
            # 过滤完成的 trial
            df_complete = df[df["state"] == "COMPLETE"].copy()

            # 按目标方向选前N名
            if str(study.direction.name).upper() == "MAXIMIZE":
                top = df_complete.nlargest(read_lines_num, "value")
            else:
                raise NotImplementedError("目前仅支持最大化目标")

            # 另存成一个 Top-N 文件；避免与全量结果冲突
            Path(self.cfg.tune.result_pth).mkdir(parents=True, exist_ok=True)
            top_path = f"{self.cfg.tune.result_pth}/{self.cfg.tune.task}_top{read_lines_num}_seed={self.cfg.tune.seed_base}.xlsx"
            top.to_excel(top_path, index=False)

            # 基本验证
            assert Path(top_path).exists(), f"Top-N 写入失败：{top_path} 不存在"
            # 可选：再读一次并校验行数
            # check_top = pd.read_excel(top_path, engine="openpyxl")
            # assert len(check_top) <= read_lines_num, "Top-N 文件行数异常"

            logging.info(f"Top-N 最优结果已保存：{top_path}")
        except Exception as e:
            logging.error(f"保存 Top-N 失败：{e}")
@hydra.main(config_path='../../config', config_name='config')
def main(cfg: DictConfig) -> None:

    setup_logging()
    if cfg.train_enabled:
        target_dir = os.path.join(cfg.dir_path, cfg.best_model_save_path)

        # 2. 依然要记得先创建文件夹！(这一步不能省)
        os.makedirs(target_dir, exist_ok=True)

        # 3. 保存配置
        save_path = os.path.join(target_dir, "run_config.yaml")
        with open(save_path, 'w') as f:
            OmegaConf.save(config=cfg, f=f, resolve=True)

        print(f"[Info] 配置文件已保存至: {save_path}")
        general_trainer = General_Trainer(cfg=cfg)
        general_trainer.search_seed()
        general_trainer.training_env= general_trainer.gen_training_env(env_yaml=cfg.train_conf)
        general_trainer.initialize_train_env()
        general_trainer.eval_env = general_trainer.gen_eval_env(env_yaml=cfg.eval_conf)
        general_trainer.initialize_eval_env()
        general_trainer.gen_model()
        general_trainer.learn()

        return
    elif cfg.get('tune') and cfg.tune.get('enabled',False):
        tuner = General_Tuner(cfg)
        study = tuner.optuna_study()
        print("[Optuna] Best trial:")
        print("  number=", study.best_trial.number)
        print("  value=", study.best_trial.value)
        print("  params=", study.best_trial.params)
        return
    elif cfg.get('read_db') and cfg.read_db.get('enabled', False):
        reader = General_Reader(cfg)
        reader.read_db(cfg.read_db.db_path, study_name=cfg.read_db.study_name,
                       read_lines_num=cfg.read_db.read_lines_num)
        return  # read_db 模式结束后直接返回，不走普通训练

    elif cfg.get('test') and cfg.test.get('enabled',False):
        general_tester = General_Tester(cfg=cfg)
        general_tester.search_seed()
        results = {
            'Type': [],
            'rl_3D_err': [],
            'or_3D_err': [],
            'rl_ll_err': [],
            'or_ll_err': [],
            'rl_3D_err_RMSE': [],
            'or_3D_err_RMSE': [],
            'rl_ll_err_RMSE': [],
            'or_ll_err_RMSE': [],
        }
        for test_type in cfg.test.test_type_list:
            results['Type'].append(test_type)
            general_tester.collecting_features(test_type= test_type)
            general_tester.test(test_type=test_type, results=results)
        general_tester.record_results(results=results)

        return  # test 模式结束后直接返回
    else:
        raise ValueError("Invalid mode")


if __name__ == '__main__':
    main()
