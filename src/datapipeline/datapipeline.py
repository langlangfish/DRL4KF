from tqdm.contrib import itertools

from src.utils.package import Any,DictConfig, Path, pickle, pd,np,Dict


class Data_Pipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg
        self.dir_path = Path(cfg.dir_path)
        self.raw_data_path = Path('data/')

    def init_env_data(self,data:DictConfig,mode: str = "train"):
        raw_data = {
            'data_truth_dic': None,
            'losfeature': None,
            'traj_sum_df': None,
            'traj_sum_df_nocanyon': None,
        }
        for key,file_name in data.items():
            raw_data[key] = self.get_new_data(file_name=file_name)
        if mode == "train":
            exclude_data = self.cfg.exclude_data
        elif mode == "eval":
             exclude_data = self.cfg.exclude_data
        elif mode == "test":
             exclude_data = self.cfg.exclude_data
        else:
            raise ValueError("mode must be 'train' or 'test'")

        for name, file_list in exclude_data.items():
            file_list = set(file_list)
            if name == 'exclude_canyon':
                raw_data['traj_sum_df_nocanyon'] = raw_data['traj_sum_df_nocanyon'][
                    ~raw_data['traj_sum_df_nocanyon']['tripID'].isin(file_list)]
            elif name == 'tripID_to_remove':
                raw_data['traj_sum_df'] = raw_data['traj_sum_df'][~raw_data['traj_sum_df']['tripID'].isin(file_list)]
                raw_data['traj_sum_df_nocanyon'] = raw_data['traj_sum_df_nocanyon'][~raw_data['traj_sum_df_nocanyon']['tripID'].isin(file_list)]
            elif name == 'without_full_traj':
                raw_data['without_0708'] = raw_data['traj_sum_df'][~raw_data['traj_sum_df']['tripID'].isin(file_list)].reset_index(drop=True)
                raw_data['without_0708']['Unnamed: 0'] = range(len(raw_data['without_0708']))
                raw_data['0708'] = raw_data['traj_sum_df'][raw_data['traj_sum_df']['tripID'].isin(file_list)].reset_index(drop=True)
                raw_data['0708']['Unnamed: 0'] = range(len(raw_data['0708']))

        env_data = dict(
            data_truth_dic=raw_data['data_truth_dic'],
            traj_sum_df=raw_data['traj_sum_df'],
            losfeature=raw_data['losfeature'],
            traj_openroad=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'openroad')][
                'tripID'].values.tolist(),
            traj_canyon=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'canyon')][
                'tripID'].values.tolist(),
            traj_highway=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'highway')][
                'tripID'].values.tolist(),
            traj_forest=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'forest')][
                'tripID'].values.tolist(),
            traj_overpass=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'overpass')][
                'tripID'].values.tolist(),
            traj_full_nocanyon=raw_data['traj_sum_df_nocanyon']['tripID'].values.tolist(),
            traj_full=raw_data['traj_sum_df']['tripID'].values.tolist(),
            traj_without_0708 = raw_data['without_0708']['tripID'].values.tolist(),
            traj_0708 = raw_data['0708']['tripID'].values.tolist(),
        )

        env_data_0708 = dict (
            data_truth_dic=raw_data['data_truth_dic'],
            traj_sum_df=raw_data['traj_sum_df'],
            losfeature=raw_data['losfeature'],
            traj_without_0708=raw_data['without_0708']['tripID'].values.tolist(),
            traj_0708=raw_data['0708']['tripID'].values.tolist(),
            traj_openroad=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'openroad')][
                'tripID'].values.tolist(),
            traj_canyon=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'canyon')][
                'tripID'].values.tolist(),
            traj_highway=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'highway')][
                'tripID'].values.tolist(),
            traj_forest=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'forest')][
                'tripID'].values.tolist(),
            traj_overpass=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'overpass')][
                'tripID'].values.tolist(),
        )
        # for tripID, episode in env_data['data_truth_dic'].items():
        #     env_data['data_truth_dic'][tripID] = episode.reset_index(drop=True)
        #     df = env_data['data_truth_dic'][tripID]
        #     needed_cols = ['X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
        #                    'CN0_mean', 'EA_mean', 'PR_mean', 'satnum', 'velocity', 'VX_RLpredict', 'VY_RLpredict',
        #                    'VZ_RLpredict']
        #     for c in needed_cols:
        #         if c not in df.columns:
        #             df[c] = np.nan
        if mode == 'train' or mode == 'eval':
            return env_data
        else:
            return env_data_0708


    def init_env_data_traj_all(self,data:DictConfig,mode: str = "train"):
        raw_data = {
            'data_truth_dic': None,
            'losfeature': None,
            'traj_sum_df': None,
            'traj_sum_df_nocanyon': None,
        }
        for key,file_name in data.items():
            raw_data[key] = self.get_new_data(file_name=file_name)
        if mode == "train":
            exclude_data = self.cfg.exclude_data
        elif mode == "eval":
             exclude_data = self.cfg.exclude_data
        elif mode == "test":
             exclude_data = self.cfg.exclude_data
        else:
            raise ValueError("mode must be 'train' or 'test'")

        for name, file_list in exclude_data.items():
            file_list = set(file_list)
            if name == 'exclude_canyon':
                raw_data['traj_sum_df_nocanyon'] = raw_data['traj_sum_df_nocanyon'][
                    ~raw_data['traj_sum_df_nocanyon']['tripID'].isin(file_list)]
            elif name == 'tripID_to_remove':
                raw_data['traj_sum_df'] = raw_data['traj_sum_df'][~raw_data['traj_sum_df']['tripID'].isin(file_list)]
                raw_data['traj_sum_df_nocanyon'] = raw_data['traj_sum_df_nocanyon'][~raw_data['traj_sum_df_nocanyon']['tripID'].isin(file_list)]
            elif name == 'without_full_traj':
                raw_data['without_0708'] = raw_data['traj_sum_df'][~raw_data['traj_sum_df']['tripID'].isin(file_list)].reset_index(drop=True)
                raw_data['without_0708']['Unnamed: 0'] = range(len(raw_data['without_0708']))
                raw_data['0708'] = raw_data['traj_sum_df'][raw_data['traj_sum_df']['tripID'].isin(file_list)].reset_index(drop=True)
                raw_data['0708']['Unnamed: 0'] = range(len(raw_data['0708']))

        env_data = dict(
            data_truth_dic=raw_data['data_truth_dic'],
            traj_sum_df=raw_data['traj_sum_df'],
            losfeature=raw_data['losfeature'],
            traj_openroad=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'openroad')][
                'tripID'].values.tolist(),
            traj_canyon=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'canyon')][
                'tripID'].values.tolist(),
            traj_highway=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'highway')][
                'tripID'].values.tolist(),
            traj_forest=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'forest')][
                'tripID'].values.tolist(),
            traj_overpass=raw_data['traj_sum_df'].loc[(raw_data['traj_sum_df']['Type'] == 'overpass')][
                'tripID'].values.tolist(),
            traj_full_nocanyon=raw_data['traj_sum_df_nocanyon']['tripID'].values.tolist(),
            traj_full=raw_data['traj_sum_df']['tripID'].values.tolist(),
            traj_without_0708 = raw_data['without_0708']['tripID'].values.tolist(),
            traj_0708 = raw_data['0708']['tripID'].values.tolist(),
        )

        # env_data_0708 = dict (
        #     data_truth_dic=raw_data['data_truth_dic'],
        #     traj_sum_df=raw_data['traj_sum_df'],
        #     losfeature=raw_data['losfeature'],
        #     traj_without_0708=raw_data['without_0708']['tripID'].values.tolist(),
        #     traj_0708=raw_data['0708']['tripID'].values.tolist(),
        #     traj_openroad=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'openroad')][
        #         'tripID'].values.tolist(),
        #     traj_canyon=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'canyon')][
        #         'tripID'].values.tolist(),
        #     traj_highway=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'highway')][
        #         'tripID'].values.tolist(),
        #     traj_forest=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'forest')][
        #         'tripID'].values.tolist(),
        #     traj_overpass=raw_data['0708'].loc[(raw_data['0708']['Type'] == 'overpass')][
        #         'tripID'].values.tolist(),
        # )
        # for tripID, episode in env_data['data_truth_dic'].items():
        #     env_data['data_truth_dic'][tripID] = episode.reset_index(drop=True)
        #     df = env_data['data_truth_dic'][tripID]
        #     needed_cols = ['X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
        #                    'CN0_mean', 'EA_mean', 'PR_mean', 'satnum', 'velocity', 'VX_RLpredict', 'VY_RLpredict',
        #                    'VZ_RLpredict']
        #     for c in needed_cols:
        #         if c not in df.columns:
        #             df[c] = np.nan
        return env_data



    def get_new_data(self,file_name:str)-> Any:
        """
        根据文件名后缀读取 .pkl 或 .csv 文件。

        Args:
            file_name (str): 要读取的文件名 (例如 'my_data.csv').

        Returns:
            Any: 如果是.csv文件, 返回pandas DataFrame;
                 如果是.pkl文件, 返回解包后的Python对象;
                 如果文件类型不支持或文件不存在，将引发异常。
        """
        file_path = self.dir_path / self.raw_data_path / file_name

        print(f"正在尝试读取文件: {file_path}")

        # 1. 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"错误: 文件 '{file_path}' 不存在。")

        # 2. 根据文件后缀选择读取方式
        file_suffix = file_path.suffix.lower()  # 使用.lower()来兼容.CSV等大写后缀

        if file_suffix == '.pkl':
            try:
                with open(file_path, 'rb') as f:  # 'rb' 表示 'read binary'，读取pickle文件必须用二进制模式
                    data = pickle.load(f)
                    data_1 = {k: v for i, (k, v) in enumerate(data.items()) if i > 31}
                print("成功读取 Pickle 文件。")
                return data_1
            except (pickle.UnpicklingError, EOFError) as e:
                raise IOError(f"无法读取 Pickle 文件 '{file_path}': {e}")
        elif file_suffix == '.csv':
            try:
                data = pd.read_csv(file_path)
                data_2 = data.iloc[32:].reset_index(drop=True)
                data_2['Unnamed: 0'] = range(len(data_2))
                print("成功读取 CSV 文件。")
                return data_2
            except Exception as e:
                raise IOError(f"无法读取 CSV 文件 '{file_path}': {e}")
        else:
            # 3. 如果文件类型不支持，抛出异常
            raise ValueError(f"不支持的文件类型: '{file_suffix}'。只支持 '.pkl' 和 '.csv'。")

