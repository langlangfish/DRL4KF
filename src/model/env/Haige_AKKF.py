from src.utils.package import gym, spaces, np, random,logging
from src.utils.env_param_new import *
from src.utils.funcs.utilis import dist_err_XYZ,haversine
from src.utils.gnss_lib import coordinates as coord


class GPSPosition_continuous_lospos_KG_onlyKtNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, env_data, cfg, traj_type, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, kg_action_scale, conv_corr, allcorrect,
                 selfvel, traj_ratio, tripid_range_ratio,step_print):
        super(GPSPosition_continuous_lospos_KG_onlyKtNallcorrect, self).__init__()
        self.rollout_rl_error_sum = 0
        self.rollout_step_count = 0
        self.rollout_bl_error_sum = 0
        self.step_print = step_print
        self.sigma_mahalanobis = sigma_mahalanobis
        self.outlier_velocity = None
        self.Kalman_gain = np.zeros((3, 3))
        self.max_visible_sat = 24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        self.cfg = cfg
        self.conv_corr = conv_corr
        self.selfvel = selfvel
        self.allcorrect = allcorrect
        self.env_data = env_data
        self.trajID_list = env_data[traj_type]
        self.traj_type = traj_type
        self.traj_ratio = [traj_ratio[0], traj_ratio[1] * tripid_range_ratio]
        self.trajdata_range = trajdata_range
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
             'pos': spaces.Box(low=-1, high=1, shape=(1, (self.pos_num - 1) * 3), dtype=np.float)})

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.kg_diagonal = kg_action_scale['diagonal']
        self.kg_else = kg_action_scale['else']
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 12),
                                       dtype=np.float)
        self.total_reward = 0
        self.baseline_model = baseline_mod
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        if self.trajdata_sort == 'sorted':
            self.traj_idx = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])

        self.data_truth_dic = self.env_data['data_truth_dic'].copy()
        #=========new reward
        self.enable_shaping = False  # bool 标志：是否启用奖励塑形
        self.shaping_weight = 0.1  # 塑形奖励的权重
        self.last_error_rl = None  # 存储上一步的误差 (3D)
        self.last_llerr_rl = None  # 存储上一步的误差 (2D)
        # 2. 奖励缩放
        self.reward_scale = 0.2
    def reset_rollout_metrics(self):
        """
        这个函数不是给 gym 用的，是给 Callback 在 rollout 开始时强制调用的
        """
        self.rollout_rl_error_sum = 0.0
        self.rollout_bl_error_sum = 0.0
        self.rollout_step_count = 0

    def _select_traj(self):

        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.traj_idx = self.traj_idx + 1
            if self.traj_idx > self.trajdata_range[1]:
                self.traj_idx = self.trajdata_range[0]

    def _load_trajdata_init_kf_velocity(self):
        self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]] = self.env_data['data_truth_dic'][
            self.trajID_list[self.traj_idx]].copy().reset_index(drop=True)
        self.baseline = self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].copy()
        self.losfeature = self.env_data['losfeature'][self.trajID_list[self.traj_idx]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3)  # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime']  # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values) - 1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_model:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_model == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_ratio[0])
        if self.traj_ratio[0] > 0:  # 只要剩下部分轨迹的定位结果
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['X_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Y_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Z_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['X_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])
        

    def _normalize_pos(self, state):
        # state[0] = (state[0] - xecef_min) / (xecef_max-xecef_min)
        # state[1] = (state[1] - yecef_min) / (yecef_max-yecef_min)
        # state[2] = (state[2] - zecef_min) / (zecef_max-zecef_min)
        # state[0] = (state[0] - np.mean(state[0])) / np.std(state[0])
        # state[1] = (state[1] - np.mean(state[1])) / np.std(state[1])
        # state[2] = (state[2] - np.mean(state[2])) / np.std(state[2])
        state[0] = state[0] / 20
        state[1] = state[1] / 20
        state[2] = state[2] / 20
        return state



    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:, 1] = (gnss[:, 1]) / res_max
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _obs_padding(self, obs):
        # --- 填充数组以确保尺寸正确 ---
        expected_cols = self.pos_num - 1  # 期望的列数为4
        current_cols = obs.shape[1]  # 当前数组的列数
        # 如果列数不足，则进行填充
        if current_cols < expected_cols:
            # 计算需要填充的列数
            padding_cols = expected_cols - current_cols
            # 创建一个需要填充的零数组
            padding = np.zeros((3, padding_cols))
            # 将原始数组和零数组拼接在一起
            obs = np.concatenate((obs, padding), axis=1)
        # --- 填充结束 ---
        return obs

    def _process_gnss_feature(self):
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp = self.losfeature[self.datatime[self.cur_idx]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([self.max_visible_sat, self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5]  # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat, :]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp
        return obs_feature

    def _next_observation(self):

        obs = np.array([
            self.baseline.loc[self.current_step: self.pre_idx, 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Z_RLpredict'].values])
        # vel = np.array([
        #     self.baseline.loc[self.current_step: self.pre_idx, 'VX_RLpredict'].values,
        #     self.baseline.loc[self.current_step: self.pre_idx, 'VY_RLpredict'].values,
        #     self.baseline.loc[self.current_step: self.pre_idx, 'VZ_RLpredict'].values])
        if 'spp' in self.baseline_model:
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_spp']]],
                            axis=1)
            # vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_wls']],
            #                       [self.baseline.loc[self.cur_idx, 'VYEcefMeters_wls']],
            #                       [self.baseline.loc[self.cur_idx, 'VZEcefMeters_wls']]],
            #                 axis=1)
        elif self.baseline_model == 'rtk':
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_rtk']]],
                            axis=1)
            # vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_rtk']],
            #                       [self.baseline.loc[self.cur_idx, 'VYEcefMeters_rtk']],
            #                       [self.baseline.loc[self.cur_idx, 'VZEcefMeters_rtk']]],
            #                 axis=1)
        obs = np.diff(obs, n=1, axis = -1)  # 位置差分
        # vel = np.diff(vel,n=1,axis = -1)
        obs = self._obs_padding(obs)
        # vel = self._obs_padding(vel)
        obs = self._normalize_pos(obs)
        # vel = self._normalize_vel(vel)
        obs_feature = self._process_gnss_feature()
        obs_all = {'pos': obs.reshape(1, 3 * (self.pos_num - 1), order='F'),
                   'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   }
        self.obs = obs_all
        return obs_all



    def _update_idx(self):
        self.max_index = len(self.baseline) - 1
        self.cur_idx = min(self.current_step + (self.pos_num - 1), self.max_index)
        self.pre_idx = min(self.current_step + (self.pos_num - 2), self.max_index)

    def _check_done(self):
        return self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_ratio[-1]\
            - self.pos_num - outlayer_in_end_ecef
    def _parse_action(self, action):
        action = np.reshape(action, [1, 12])
        for idx in [0, 4, 8]:
            action[0, idx] *= self.kg_diagonal
        for idx in [1,2,3,5,6,7]:
            action[0, idx] *= self.kg_else
        predict_kg = np.array([[action[0, 0], action[0, 1], action[0, 2]],
                               [action[0, 3], action[0, 4], action[0, 5]],
                               [action[0, 6], action[0, 7], action[0, 8]]])
        delta_pos = action[0, 9:] * self.continuous_action_scale
        return predict_kg, delta_pos
    def _get_xyz(self,method):
        if method == 'gt':
            return np.array([
                self.baseline.loc[self.cur_idx, f'ecefX_{method}'],
                self.baseline.loc[self.cur_idx, f'ecefY_{method}'],
                self.baseline.loc[self.cur_idx, f'ecefZ_{method}']
            ])
        else:
            return np.array([
                self.baseline.loc[self.cur_idx,f'XEcefMeters_{method}'],
                self.baseline.loc[self.cur_idx, f'YEcefMeters_{method}'],
                self.baseline.loc[self.cur_idx, f'ZEcefMeters_{method}']
            ])
    def _get_velocity(self):
        if self.selfvel:
            return np.array([
                self.baseline.loc[self.cur_idx, f'VXEcefMeters_self'],
                self.baseline.loc[self.cur_idx, f'VYEcefMeters_self'],
                self.baseline.loc[self.cur_idx, f'VZEcefMeters_self']
            ])
        else:
            return np.array([
                self.baseline.loc[self.cur_idx, f'VXEcefMeters_wls'],
                self.baseline.loc[self.cur_idx, f'VYEcefMeters_wls'],
                self.baseline.loc[self.cur_idx, f'VZEcefMeters_wls']
            ])
    def _check_velocity_outlier(self,velocity):
        if np.sqrt(np.sum(velocity**2)) > outlier_velocity:
            return True
        else:
            return False
    # def _kf_step_(self,state,vel,measurement,predict_kg,Q,R,utc_pre,utc_cur):
    #     F = np.eye(3)
    #     H = np.eye(3)
    #     x_predict = F @ state + vel.T *(utc_cur-utc_pre)
    #     P_predict = F @ self.P @ F.T + Q
    #     d = dist_err_XYZ(measurement,H @ x_predict)
    #     if d < self.sigma_mahalanobis:
    #         y = measurement.T - H @ x_predict
    #         S = H @ P_predict @ H.T + R
    #         if np.all(self.Kalman_gain == 0) :
    #             K = P_predict @ H.T @ np.linalg.inv(S)
    #             K += predict_kg
    #             self.Kalman_gain = K
    #         else:
    #             K = self.Kalman_gain + predict_kg
    #         x_update = x_predict + K @ y
    #         self.P = (np.eye(3) - K @ H) @ P_predict
    #         self.covx = R
    #     else:
    #         self.P = P_predict + 100 * Q
    #         x_update = x_predict
    #     return x_update

    def _kf_step(self, state, vel, measurement, predict_kg, Q, R, utc_pre, utc_cur):
        F = np.eye(3)
        H = np.eye(3)

        # 1. Prediction Step
        dt = utc_cur - utc_pre
        x_predict = F @ state + vel.T * dt
        P_predict = F @ self.P @ F.T + Q

        # 2. Measurement Update Check (Mahalanobis Gate)
        d = dist_err_XYZ(measurement, H @ x_predict)

        if d < self.sigma_mahalanobis:
            y = measurement.T - H @ x_predict  # Innovation (Residual)
            S = H @ P_predict @ H.T + R  # Innovation Covariance
            # --- 核心修改开始 --
            # 使用 np.linalg.solve 代替 inv 提高数值稳定性
            k_base = P_predict @ H.T @ np.linalg.inv(S)
            k_actual = k_base + predict_kg
            # C. 更新状态
            x_update = x_predict + k_actual @ y

            # D. 更新 P 阵 (使用 Joseph Form 以保证稳定性)
            # P = (I - KH) P (I - KH).T + K R K.T
            I = np.eye(3)
            I_KH = I - k_base @ H

            # 这是一个即使用“错误”的 K 也能保证 P 正定的公式
            self.P = I_KH @ P_predict @ I_KH.T + k_base @ R @ k_base.T

            # --- 核心修改结束 ---

            # self.covx = R  # 这里保留你的逻辑，虽然 covx 一般不这么更新

            # 保存 K 仅供记录/可视化，不参与下一次计算
            self.Kalman_gain = k_actual

        else:
            # 异常值处理
            # 注意：这里 P 膨胀后，下一帧计算出的 K_base 会自动变大，这就是 KF 的自适应性
            self.P = P_predict + 100 * Q
            x_update = x_predict

        return x_update

    def RL4KFGSDC(self,measurement,vel,predict_kg,policy=True):
        idx_cur = self.current_step + (self.pos_num - 1)
        idx_pre = self.current_step + (self.pos_num - 2)
        utc_cur = self.datatime[idx_cur]
        utc_pre = self.datatime[idx_pre]
        state = np.array([
            self.baseline.loc[idx_pre, 'X_RLpredict'],
            self.baseline.loc[idx_pre, 'Y_RLpredict'],
            self.baseline.loc[idx_pre, 'Z_RLpredict']
        ])
        self.outlier_velocity = outlier_velocity
        if self._check_velocity_outlier(vel):
            vel = self.v_pre
        Q = self.covv
        R = sigma_x ** 1.250 * np.eye(3)
        if (utc_cur - utc_pre) < interrupt_time:  # 如果时间中断，重新初始化
            state = self._kf_step(state,vel,measurement,predict_kg,Q,R,utc_pre,utc_cur)
        else:
            state = measurement
            self.P += sigma_x ** 1.25 * np.eye(3) * 100
        if not policy:
            state = np.array([
                self.baseline.loc[idx_cur, 'XEcefMeters_spp'],
                self.baseline.loc[idx_cur, 'YEcefMeters_spp'],
                self.baseline.loc[idx_cur, 'ZEcefMeters_spp']
            ])
        self.v_pre = vel
        return state
    def _update_baseline(self,rl_xyz):
        pos_map = {
            'X': 0,
            'Y': 1,
            'Z': 2,
        }
        for key, value in pos_map.items():
            self.baseline.loc[self.cur_idx, [f'{key}_RLpredict']] = rl_xyz[value]
    def _get_features(self,rl_xyz,vel):
        rl_xyz = np.asarray(rl_xyz)
        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        Velocity = np.linalg.norm(vel)
        feature_dict = {
            'X_RLpredict': rl_xyz[0],
            'Y_RLpredict': rl_xyz[1],
            'Z_RLpredict': rl_xyz[2],
            'CN0_mean': CN0_mean,
            'EA_mean': EA_mean,
            'PR_mean': PR_mean,
            'satnum': satnum,
            'velocity': Velocity
        }
        return feature_dict
    def _update_data_truth_dic(self,feature_dict):
        for key, value in feature_dict.items():
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[self.cur_idx, [key]] = value

    def _compute_reward_(self,rl_xyz,gt_xyz,obs_xyz,kf_xyz):
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_obs = np.linalg.norm(obs_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        llh1 =coord.ecef2geodetic(gt_xyz)
        llh2 =coord.ecef2geodetic(kf_xyz)
        llh3 =coord.ecef2geodetic(rl_xyz)
        llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        if self.reward_setting == 'RMSE':
            return 10 - error_rl
        elif self.reward_setting == 'RMSEadv':
            return error_obs - error_rl
        elif self.reward_setting == 'RMSEadv_kf':
            return error_kf - error_rl
        elif self.reward_setting == 'RMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6)
        elif self.reward_setting == '2dRMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6) + llerr_kf - llerr_rl
        else:
            raise ValueError(f"Unknown reward setting: {self.reward_setting}")

    def _compute_reward(self, rl_xyz, gt_xyz, obs_xyz, kf_xyz):

        # --- 1. 基础 3D 误差计算 (常用) ---
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        base_reward = 0.0
        llerr_rl = None  # 默认为空，仅在 2D 模式下计算
        # --- 2. 计算基础奖励 (根据设置) ---
        if self.reward_setting == 'RMSEadv_kf_tanh':
            # 3D 优势 (KF - RL)，使用 tanh 缩放到 [-1, 1]
            advantage = error_kf - error_rl
            base_reward = np.tanh(advantage * self.reward_scale)
        elif self.reward_setting == 'RMSEratio_kf_tanh':
            # 3D 归一化比率，使用 tanh 压缩到 [-1, 1]
            ratio = (error_kf - error_rl) / (error_kf + 1e-6)
            base_reward = np.tanh(ratio)
        elif self.reward_setting == 'RMSEmixratio_kf_tanh':
            llh1 = coord.ecef2geodetic(gt_xyz)
            llh2 = coord.ecef2geodetic(kf_xyz)
            llh3 = coord.ecef2geodetic(rl_xyz)
            llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
            llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
            # raw_2d = 5 * np.log(np.maximum((2 * llerr_kf - llerr_rl) / llerr_rl, 1e-6))
            # reward_2d = np.clip(raw_2d, -10, 10)
            # raw_3d = 5*np.log(np.maximum((2*error_kf-error_rl)/error_kf,1e-6))
            # reward_3d = np.clip(raw_3d, -10, 10)
            # ratio_3d = (error_kf - error_rl) / (error_kf + 1e-6)
            # ratio_2d = (llerr_kf-llerr_rl) / (llerr_kf + 1e-6)
            reward_2d = 10 * np.tanh(3*((llerr_kf-llerr_rl)/(llerr_kf + 1e-6)))
            reward_3d = 10 * np.tanh(3*((error_kf-error_rl)/(error_kf + 1e-6)))
            w_2d = 0.7
            w_3d = 0.3
            base_reward = w_2d * reward_2d+ w_3d * reward_3d
            if reward_2d > 0:
                base_reward = base_reward +10
            return base_reward,llerr_kf,llerr_rl

        # --- 推荐的 2D 奖励 (纯 2D 目标) ---
        elif self.reward_setting in ['2dRMSEadv_kf_tanh', '2dRMSEratio_kf_tanh']:
            # 仅在需要时执行昂贵的坐标转换
            llh1 = coord.ecef2geodetic(gt_xyz)
            llh2 = coord.ecef2geodetic(kf_xyz)
            llh3 = coord.ecef2geodetic(rl_xyz)
            llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
            llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
            if self.reward_setting == '2dRMSEadv_kf_tanh':
                # 2D 优势 (KF - RL)，使用 tanh 缩放
                advantage_2d = llerr_kf - llerr_rl
                base_reward = np.tanh(advantage_2d * self.reward_scale)
            elif self.reward_setting == '2dRMSEratio_kf_tanh':
                # 2D 归一化比率，使用 tanh 压缩
                ratio_2d = (llerr_kf - llerr_rl) / (llerr_kf + 1e-6)
                base_reward = np.tanh(ratio_2d)
        # --- 原始奖励设置 (保留对比) ---
        else:
            # 在 reset 中更新 last_error, 否则第一次 step 会报错
            if self.last_error_rl is not None:
                self.last_error_rl = error_rl
            if self.last_llerr_rl is not None and llerr_rl is not None:
                self.last_llerr_rl = llerr_rl
            raise ValueError(f"Unknown reward setting: {self.reward_setting}")
        # --- 3. 奖励塑形 (PBRS) (可选) ---
        # 奖励智能体“相对于上一刻的自己”的进步
        shaping_reward = 0.0
        if self.enable_shaping:
            # 基于主要目标（3D 或 2D）来塑形
            if '2d' in self.reward_setting:
                # 使用 2D 误差进行塑形
                if llerr_rl is None:  # 确保 2D 误差已计算
                    llh1 = coord.ecef2geodetic(gt_xyz)
                    llh3 = coord.ecef2geodetic(rl_xyz)
                    llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')

                if self.last_llerr_rl is not None:
                    shaping_reward = (self.last_llerr_rl - llerr_rl) * self.shaping_weight
                self.last_llerr_rl = llerr_rl  # 更新 2D 误差
                self.last_error_rl = error_rl  # 同时也更新 3D，以防万一

            else:
                # 默认使用 3D 误差进行塑形
                if self.last_error_rl is not None:
                    shaping_reward = (self.last_error_rl - error_rl) * self.shaping_weight
                self.last_error_rl = error_rl  # 更新 3D 误差
                # (如果需要，也可以更新 2D)
        # --- 4. 最终奖励 ---
        # 总奖励 = 基础奖励 (与 KF/OBS 对比) + 塑形奖励 (与自己对比)
        return base_reward + shaping_reward


    def _print_perstep_info(self,timestep,obs_xyz,rl_xyz,gt_xyz,reward):
        print(
            f'{self.trajID_list[self.traj_idx]}, Time {timestep}/{self.timeend} '
            f'Baseline dist: [{np.abs(obs_xyz[0] - gt_xyz[0]): .2f}, {np.abs(obs_xyz[1] - gt_xyz[1]): .2f}, {np.abs(obs_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RL dist: [{np.abs(rl_xyz[0] - gt_xyz[0]): .2f}, {np.abs(rl_xyz[1] - gt_xyz[1]): .2f}, {np.abs(rl_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RMSEadv: {reward: 0.2e} m.')


    def _get_info(self, errors, traj_ID):
        return {
            'traj_idx': self.traj_idx,
            'traj_ID': traj_ID,
            'current_step': self.current_step,
            'baseline': self.baseline,
            'error': errors['rl'],
            'error_bl': errors['bl'],
            'error_kf': errors['kf'],
            'rl2gt_err': errors['rl2gt'],
            'rl_ll_err': errors['rl_ll'],
            'bl_ll_err': errors['bl_ll']
        }

    def reset(self):
        self._select_traj()
        self._load_trajdata_init_kf_velocity()
        # self.current_step = int(np.ceil(len(self.baseline) * self.traj_ratio[0]))
        self._update_idx()

        obs = self._next_observation()
        self.total_reward = 0
        self.last_error_rl = None
        self.last_llerr_rl = None
        return obs

    def step(self, action):
        done= self._check_done()
        timestep = self.baseline.loc[self.cur_idx, 'UTCtime']
        predict_kg, delta_pos = self._parse_action(action)
        obs_xyz = self._get_xyz(method=self.baseline_model)
        kf_xyz = self._get_xyz(method='kf')
        gt_xyz = self._get_xyz(method='gt')
        vel = self._get_velocity()
        # velocity = np.sqrt(np.sum(vel_wls)**2)
        x_wls = obs_xyz + delta_pos
        rl_xyz = self.RL4KFGSDC(x_wls,vel,predict_kg,policy=True)
        self._update_baseline(rl_xyz)
        feature_dict = self._get_features(rl_xyz, vel)
        self._update_data_truth_dic(feature_dict)
        reward,llerr_kf,llerr_rl = self._compute_reward(rl_xyz,gt_xyz,obs_xyz,kf_xyz)
        self.total_reward += reward
        if self.step_print:
            self._print_perstep_info(timestep, obs_xyz, rl_xyz, gt_xyz, reward)
        self.current_step += 1
        self._update_idx()
        obs = [] if done else self._next_observation()
        self.rollout_rl_error_sum += llerr_rl
        self.rollout_bl_error_sum += llerr_kf
        self.rollout_step_count += 1
        errors = {
            'rl': np.linalg.norm(rl_xyz - gt_xyz),
            'bl': np.linalg.norm(obs_xyz - gt_xyz),
            'kf': np.linalg.norm(kf_xyz - gt_xyz),
            'rl2gt': np.sum((rl_xyz - gt_xyz) ** 2),
            'rl_ll': llerr_rl,
            'bl_ll': llerr_kf
        }
        info = self._get_info(errors, self.trajID_list[self.traj_idx])
        # 防止除以0 (虽然理论上不可能，加个 max 安全一点)
        steps = max(1, self.rollout_step_count)

        # 1. 计算本回合的平均误差
        avg_rl_error = self.rollout_rl_error_sum / steps
        avg_bl_error = self.rollout_bl_error_sum / steps

        # 2. 将平均值写入 info 字典
        # 这些 key (episode_mean_rl_error) 就是你在 Callback 里要读取的名字
        info['episode_mean_rl_error'] = avg_rl_error
        info['episode_mean_bl_error'] = avg_bl_error

        # (可选) 如果你想在 log 里看到这一回合走了多少步
        info['episode_steps'] = steps
        return obs, reward, done, info

class Kt_Withvel(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, env_data, cfg, traj_type, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, kg_action_scale, conv_corr, allcorrect,
                 selfvel, traj_ratio, tripid_range_ratio,step_print,Q_static):
        super(Kt_Withvel, self).__init__()
        self.rollout_rl_error_sum = 0
        self.rollout_step_count = 0
        self.rollout_bl_error_sum = 0
        self.state = None
        self.step_print = step_print
        self.sigma_mahalanobis = sigma_mahalanobis
        self.outlier_velocity = outlier_velocity
        self.max_visible_sat = 13
        self.feature_num = CN0_num
        self.pos_num = traj_len
        self.cfg = cfg
        self.conv_corr = conv_corr
        self.selfvel = selfvel
        self.allcorrect = allcorrect
        self.env_data = env_data
        self.trajID_list = env_data[traj_type]
        self.traj_type = traj_type
        self.traj_ratio = [traj_ratio[0], traj_ratio[1] * tripid_range_ratio]
        self.trajdata_range = trajdata_range
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
             'pos': spaces.Box(low=-1, high=1, shape=(1, (self.pos_num - 1) * 3), dtype=np.float),
             'vel':spaces.Box(low=-1, high=1, shape=(1, 3 * (self.pos_num - 1)), dtype=np.float)})

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.kg_diagonal = kg_action_scale['diagonal']
        self.kg_up = kg_action_scale['up']
        self.kg_down = kg_action_scale['down']
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 12+3+3),
                                       dtype=np.float)
        self.total_reward = 0
        self.baseline_model = baseline_mod
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        if self.trajdata_sort == 'sorted':
            self.traj_idx = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])

        self.data_truth_dic = self.env_data['data_truth_dic'].copy()
        self.verbose = 2
        self.Q_static = Q_static
        #=========new reward
        self.enable_shaping = False  # bool 标志：是否启用奖励塑形
        self.shaping_weight = 0.1  # 塑形奖励的权重
        self.last_error_rl = None  # 存储上一步的误差 (3D)
        self.last_llerr_rl = None  # 存储上一步的误差 (2D)
        # 2. 奖励缩放
        self.reward_scale = 0.2

    def reset_rollout_metrics(self):
        """
        这个函数不是给 gym 用的，是给 Callback 在 rollout 开始时强制调用的
        """
        self.rollout_rl_error_sum = 0.0
        self.rollout_bl_error_sum = 0.0
        self.rollout_step_count = 0
    def _select_traj(self):
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.traj_idx = self.traj_idx + 1
            if self.traj_idx > self.trajdata_range[1]:
                self.traj_idx = self.trajdata_range[0]

    def _load_trajdata_init_kf_velocity(self):
        if 'spp' in self.baseline_model:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_model == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']
        self.baseline['VX_RLpredict'] = self.baseline['VXEcefMeters_wls']
        self.baseline['VY_RLpredict'] = self.baseline['VYEcefMeters_wls']
        self.baseline['VZ_RLpredict'] = self.baseline['VZEcefMeters_wls']

        if self.traj_ratio[0] > 0:  # 只要剩下部分轨迹的定位结果
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['X_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Y_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Z_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VX_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VY_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VZ_RLpredict']] = None


            self.baseline.loc[0:self.current_step - 1, ['X_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Z_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VX_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VY_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VZ_RLpredict']] = None

        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])
        # set parameter for KF
        p0 = initial_P
        v0 = initial_V
        variances = [p0**2] * 3 + [v0**2] * 3
        self.P = np.diag(variances)  # 协方差矩阵
        self.sigma_m_pos = sigma_m_vel
        self.sigma_m_vel = sigma_m_pos
        variances_QR = [self.sigma_m_pos ** 2] * 3 + [self.sigma_m_vel ** 2] * 3
        self.Q_initial = np.diag(variances_QR)
        self.R = np.diag(variances_QR) # Measurement noise
        self.datatime = self.baseline['UTCtime']  # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values) - 1, 'UTCtime']
        self.sigma_a =sigma_a

        obs_pos = self._get_position(self.baseline_model, self.baseline, self.pre_idx)
        wls_vel = self._get_velocity(self.baseline, self.pre_idx, self.selfvel)
        self.state = np.hstack([obs_pos,wls_vel])

    def _normalize_pos(self, state):
        # state[0] = (state[0] - xecef_min) / (xecef_max-xecef_min)
        # state[1] = (state[1] - yecef_min) / (yecef_max-yecef_min)
        # state[2] = (state[2] - zecef_min) / (zecef_max-zecef_min)
        # state[0] = (state[0] - np.mean(state[0])) / np.std(state[0])
        # state[1] = (state[1] - np.mean(state[1])) / np.std(state[1])
        # state[2] = (state[2] - np.mean(state[2])) / np.std(state[2])
        state[0] = state[0] / 20
        state[1] = state[1] / 20
        state[2] = state[2] / 20
        return state

    def _normalize_vel(self, state):
        state[0] = state[0] / outlier_velocity
        state[1] = state[1] / outlier_velocity
        state[2] = state[2] / outlier_velocity
        return state

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:, 1] = (gnss[:, 1]) / res_max
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _obs_padding(self, obs):
        # --- 填充数组以确保尺寸正确 ---
        expected_cols = self.pos_num - 1  # 期望的列数为4
        current_cols = obs.shape[1]  # 当前数组的列数
        # 如果列数不足，则进行填充
        if current_cols < expected_cols:
            # 计算需要填充的列数
            padding_cols = expected_cols - current_cols
            # 创建一个需要填充的零数组
            padding = np.zeros((3, padding_cols))
            # 将原始数组和零数组拼接在一起
            obs = np.concatenate((obs, padding), axis=1)
        # --- 填充结束 ---
        return obs

    def _process_gnss_feature(self):
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp = self.losfeature[self.datatime[self.cur_idx]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([self.max_visible_sat, self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5]  # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat, :]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp
        return obs_feature

    def _next_observation(self):

        obs = np.array([
            self.baseline.loc[self.current_step: self.pre_idx, 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Z_RLpredict'].values])
        vel = np.array([
            self.baseline.loc[self.current_step: self.pre_idx, 'VX_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'VY_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'VZ_RLpredict'].values])
        if 'spp' in self.baseline_model:
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_spp']]],
                            axis=1)
            vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_wls']],
                                  [self.baseline.loc[self.cur_idx, 'VYEcefMeters_wls']],
                                  [self.baseline.loc[self.cur_idx, 'VZEcefMeters_wls']]],
                            axis=1)
        elif self.baseline_model == 'rtk':
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_rtk']]],
                            axis=1)
            vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'VYEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'VZEcefMeters_rtk']]],
                            axis=1)
        obs = np.diff(obs, n=1, axis = -1)  # 位置差分
        vel = np.diff(vel,n=1,axis = -1)
        obs = self._obs_padding(obs)
        vel = self._obs_padding(vel)
        obs = self._normalize_pos(obs)
        vel = self._normalize_vel(vel)
        obs_feature = self._process_gnss_feature()
        obs_all = {'pos': obs.reshape(1, 3 * (self.pos_num - 1), order='F'),
                   'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'vel':vel.reshape(1, 3 * (self.pos_num - 1), order='F')}
        self.obs = obs_all
        return obs_all

    def _update_idx(self):
        self.max_index = len(self.baseline) - 1
        self.cur_idx = int(min(self.current_step + (self.pos_num - 1), self.max_index))
        self.pre_idx = int(min(self.current_step + (self.pos_num - 2), self.max_index))
    @staticmethod
    def _initialize_current_step(baseline,traj_ratio):
        return int(np.ceil(len(baseline) * traj_ratio[0]))

    def _check_done(self):
        return self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_ratio[-1]\
            - self.pos_num - outlayer_in_end_ecef

    def _parse_action(self, action):
        action = action.reshape(1, -1)

        # fill_indices = np.array([0, 3, 7, 10, 14, 17, 18, 21, 25, 28, 32, 35])
        diag_indices_set = {0, 7, 14, 21, 28, 35}
        up_indices_set={3,10,17}
        down_indices_set={18,25,32}

        fill_indices = np.sort(list(diag_indices_set | up_indices_set | down_indices_set))

        # 2. 根据索引所属的集合选择对应的 scale
        current_scales = np.array([
            self.kg_diagonal if idx in diag_indices_set
            else self.kg_up if idx in up_indices_set
            else self.kg_down
            for idx in fill_indices
        ])

        matrix_values = action[0, :12]
        scaled_values = matrix_values * current_scales

        predict_kg_flat = np.zeros(36)
        predict_kg_flat[fill_indices] = scaled_values
        predict_kg = predict_kg_flat.reshape(6, 6)

        delta_pos = action[0, 12:15] * self.continuous_action_scale
        delta_vel = action[0, 15:18] * self.continuous_action_scale * 0.5
        # delta_vel = action[0, 15:18] * 0

        return predict_kg, delta_pos, delta_vel
    @staticmethod
    def _get_position(source, baseline, cur_idx):
        keyword = {
            'gt': {
                'X': 'ecefX_gt',
                'Y': 'ecefY_gt',
                'Z': 'ecefZ_gt'
            },
            'kf': {
                'X': 'XEcefMeters_kf',
                'Y': 'YEcefMeters_kf',
                'Z': 'ZEcefMeters_kf'
            },
            'spp': {
                'X': 'XEcefMeters_spp',
                'Y': 'YEcefMeters_spp',
                'Z': 'ZEcefMeters_spp'
            },
            'rtk': {
                'X': 'XEcefMeters_rtk',
                'Y': 'YEcefMeters_rtk',
                'Z': 'ZEcefMeters_rtk'
            }
        }[source]

        row = baseline.iloc[cur_idx]  # <— iloc
        return np.array([row[keyword['X']], row[keyword['Y']], row[keyword['Z']]])
    @staticmethod
    def _get_velocity(baseline, cur_idx, _):
        row = baseline.iloc[cur_idx]  # <— iloc
        vel = np.array([row['VXEcefMeters_wls'], row['VYEcefMeters_wls'], row['VZEcefMeters_wls']])
        return vel
    # def _get_xyz(self,method):
    #     if method == 'gt':
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'ecefX_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ecefY_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ecefZ_{method}']
    #         ])
    #     else:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx,f'XEcefMeters_{method}'],
    #             self.baseline.loc[self.cur_idx, f'YEcefMeters_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ZEcefMeters_{method}']
    #         ])
    # def _get_velocity(self):
    #     if self.selfvel:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'VXEcefMeters_self'],
    #             self.baseline.loc[self.cur_idx, f'VYEcefMeters_self'],
    #             self.baseline.loc[self.cur_idx, f'VZEcefMeters_self']
    #         ])
    #     else:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'VXEcefMeters_wls'],
    #             self.baseline.loc[self.cur_idx, f'VYEcefMeters_wls'],
    #             self.baseline.loc[self.cur_idx, f'VZEcefMeters_wls']
    #         ])
    def _check_velocity_outlier(self,velocity):
        if np.sqrt(np.sum(velocity**2)) > outlier_velocity:
            return True
        else:
            return False
    # def _kf_step_(self,state,vel,measurement,predict_kg,Q,R,utc_pre,utc_cur):
    #     F = np.eye(3)
    #     H = np.eye(3)
    #     x_predict = F @ state + vel.T *(utc_cur-utc_pre)
    #     P_predict = F @ self.P @ F.T + Q
    #     d = dist_err_XYZ(measurement,H @ x_predict)
    #     if d < self.sigma_mahalanobis:
    #         y = measurement.T - H @ x_predict
    #         S = H @ P_predict @ H.T + R
    #         if np.all(self.Kalman_gain == 0) :
    #             K = P_predict @ H.T @ np.linalg.inv(S)
    #             K += predict_kg
    #             self.Kalman_gain = K
    #         else:
    #             K = self.Kalman_gain + predict_kg
    #         x_update = x_predict + K @ y
    #         self.P = (np.eye(3) - K @ H) @ P_predict
    #         self.covx = R
    #     else:
    #         self.P = P_predict + 100 * Q
    #         x_update = x_predict
    #     return x_update

    def _kf_step(self, pos_rl, vel_rl,F,H,predict_kg, Q, R, utc_pre, utc_cur):
        # 1. Prediction Step
        z = np.hstack([pos_rl,vel_rl])
        x_predict = F @ self.state
        P_predict = F @ self.P @ F.T + Q
        y = z.T - H @ x_predict
        S = H @ P_predict @ H.T + R
        # 2. Measurement Update Check (Mahalanobis Gate)
        try:
            d = dist_err_XYZ(z[:3], (H @ x_predict)[:3])
            Sinv = np.linalg.inv(S)
            whiten = float(np.sqrt(y.T @ Sinv @ y))
        except np.linalg.LinAlgError:
            dist_err_XYZ(z[:3], (H @ x_predict)[:3])
            whiten = np.inf
            if self.verbose >= 2:
                logging.warning(f'Mahalanobis distance is infinite, utc={utc_cur}, utc_pre={utc_pre}')

        if np.isfinite(whiten) and (whiten <= self.sigma_mahalanobis):
            K = P_predict @ H.T @ np.linalg.inv(S)  # (6,6)
            K_actual = K + predict_kg
            self.state = x_predict + K_actual @ y
            self.P = (np.eye(6) - K @ H) @ P_predict
        else:
            if self.verbose >= 2:
                logging.warning(f'distance {d=} {whiten=}')
            self.state = x_predict
            self.P = P_predict + 100.0 * np.block([
                [(self.sigma_m_pos ** 2) * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), self.sigma_m_vel ** 2 * np.eye(3)]
            ])


    def _deal_with_dt_outline(self, dt, pos_rl, vel_rl, utc_cur, utc_pre):
        # 处理时间异常点，时间中断：位置对齐观测，增大不确定性
        if dt > interrupt_time or np.isnan(vel_rl).any():  # 增大噪声，用上一步预测速度替换当前速度
            # self.P += 100.0 * np.block([
            #     [(self.sigma_m_pos ** 2) * np.eye(3), np.zeros((3, 3))],
            #     [np.zeros((3, 3)), self.sigma_m_vel ** 2 * np.eye(3)]
            # ])
            vel_rl = (self.state[3:] + vel_rl) / 2
            if dt > interrupt_time:
                if self.verbose >= 2:
                    logging.warning(f'time jump {dt}s, utc={utc_cur}, utc_pre={utc_pre},'
                                    f'vel_cur={self._get_velocity(self.baseline,self.cur_idx,self.selfvel)},vel_pre={self._get_velocity(self.baseline,self.pre_idx,self.selfvel)}')
            elif np.isnan(vel_rl).any():
                if self.verbose >= 2:
                    logging.warning(f'velocity Nan')
            # self.state[:3] = pos_rl

        return pos_rl, vel_rl
    def _deal_with_vel_outline(self, vel_rl, utc_cur, utc_pre, Q):
        # 处理速度异常点
        if np.linalg.norm(vel_rl) > self.outlier_velocity:
            if self.verbose >= 2:
                logging.warning(
                    f'velocity jump {np.linalg.norm(vel_rl): .2f}m/s, utc={utc_cur}, utc_pre={utc_pre}')
            vel_rl = self.state[3:]
            self.P += 100. * Q
        return vel_rl
    
    def RL4KFGSDC(self,pos_rl,vel_rl,predict_kg,policy=True):
        idx_cur = self.current_step + (self.pos_num - 1)
        idx_pre = self.current_step + (self.pos_num - 2)
        utc_cur = self.datatime[idx_cur]
        utc_pre = self.datatime[idx_pre]
        dt = float(max(utc_cur - utc_pre, 1e-3))
        if self.Q_static:
            Q = self.Q_initial
        else:
            q11 = (dt ** 4) / 4.0
            q12 = (dt ** 3) / 2.0
            q22 = (dt ** 2)
            I3 = np.eye(3)
            Q = (self.sigma_a ** 2) * np.block([
                [q11 * I3, q12 * I3],
                [q12 * I3, q22 * I3],
            ])
        R = self.R
        F = np.eye(6, dtype=float)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        H = np.eye(6, dtype=float)
        pos_rl, vel_rl = self._deal_with_dt_outline(dt, pos_rl, vel_rl, utc_cur, utc_pre)
        vel_rl = self._deal_with_vel_outline(vel_rl, utc_cur, utc_pre, Q)

        self._kf_step(pos_rl,vel_rl,F,H,predict_kg,Q,R,utc_pre,utc_cur)
        self.v_pre = vel_rl
        rl_pos = self.state[:3].copy()
        rl_vel = self.state[3:].copy()
        if not policy:
            self.state = np.array([
                self.baseline.loc[idx_cur, 'XEcefMeters_spp'],
                self.baseline.loc[idx_cur, 'YEcefMeters_spp'],
                self.baseline.loc[idx_cur, 'ZEcefMeters_spp']
            ])
        return rl_pos,rl_vel,dt
    def _update_baseline(self,rl_xyz,rl_vel):
        pos_map = {
            'X': 0,
            'Y': 1,
            'Z': 2,
        }
        for key, value in pos_map.items():
            self.baseline.loc[self.cur_idx, [f'{key}_RLpredict']] = rl_xyz[value]
            self.baseline.loc[self.cur_idx, [f'V{key}_RLpredict']] = rl_vel[value]
    def _get_features(self,rl_pos,rl_vel):
        rl_pos = np.asarray(rl_pos)
        feature_tmp = self.losfeature[self.datatime[self.cur_idx]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        Velocity = np.linalg.norm(rl_vel)
        feature_dict = {
            'X_RLpredict': rl_pos[0],
            'Y_RLpredict': rl_pos[1],
            'Z_RLpredict': rl_pos[2],
            'CN0_mean': CN0_mean,
            'EA_mean': EA_mean,
            'PR_mean': PR_mean,
            'satnum': satnum,
            'velocity': Velocity
        }
        return feature_dict
    def _update_data_truth_dic(self,feature_dict):
        for key, value in feature_dict.items():
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[self.cur_idx, [key]] = value

    def _compute_reward_(self,rl_xyz,gt_xyz,obs_xyz,kf_xyz):
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_obs = np.linalg.norm(obs_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        llh1 =coord.ecef2geodetic(gt_xyz)
        llh2 =coord.ecef2geodetic(kf_xyz)
        llh3 =coord.ecef2geodetic(rl_xyz)
        llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        if self.reward_setting == 'RMSE':
            return 10 - error_rl
        elif self.reward_setting == 'RMSEadv':
            return error_obs - error_rl
        elif self.reward_setting == 'RMSEadv_kf':
            return error_kf - error_rl
        elif self.reward_setting == 'RMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6)
        elif self.reward_setting == '2dRMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6) + llerr_kf - llerr_rl
        else:
            raise ValueError(f"Unknown reward setting: {self.reward_setting}")

    def _compute_reward(self, rl_xyz, gt_xyz, obs_xyz, kf_xyz , pre_gt_xyz , dt,vel,rl_vel):

        # --- 1. 基础 3D 误差计算 (常用) ---
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        base_reward = 0.0
        llerr_rl = None  # 默认为空，仅在 2D 模式下计算
        # --- 2. 计算基础奖励 (根据设置) ---
        if self.reward_setting == 'RMSEadv_kf_tanh':
            # 3D 优势 (KF - RL)，使用 tanh 缩放到 [-1, 1]
            advantage = error_kf - error_rl
            base_reward = np.tanh(advantage * self.reward_scale)
        elif self.reward_setting == 'RMSEratio_kf_tanh':
            # 3D 归一化比率，使用 tanh 压缩到 [-1, 1]
            ratio = (error_kf - error_rl) / (error_kf + 1e-6)
            base_reward = np.tanh(ratio)

        elif self.reward_setting == 'RMSEmixratio_kf_tanh':

            # [位置 2D] (保留你原本的 Haversine 逻辑，准确度最高)
            llh_gt = coord.ecef2geodetic(gt_xyz)
            llh_kf = coord.ecef2geodetic(kf_xyz)
            llh_rl = coord.ecef2geodetic(rl_xyz)
            vel_gt = (gt_xyz - pre_gt_xyz) / dt
            err_2d_kf = haversine((llh_gt[0], llh_gt[1]), (llh_kf[0], llh_kf[1]), unit='m')
            err_2d_rl = haversine((llh_gt[0], llh_gt[1]), (llh_rl[0], llh_rl[1]), unit='m')
            # [速度] (假设你已保证 rl_vel, vel, vel_gt 坐标系一致且非空)
            # vel 代表基准算法(WLS/KF)的速度
            err_vel_base = np.linalg.norm(vel - vel_gt)
            err_vel_rl = np.linalg.norm(rl_vel - vel_gt)
            eps = 1e-6
            imp_2d = (err_2d_kf - err_2d_rl) / (err_2d_kf + eps)
            imp_3d = (error_kf - error_rl) / (error_kf + eps)
            imp_vel = (err_vel_base - err_vel_rl) / (err_vel_base + eps)

            r_base_2d = 10.0 * np.tanh(3.0 * imp_2d)
            r_base_3d = 10.0 * np.tanh(3.0 * imp_3d)
            r_base_vel = 10.0 * np.tanh(1.0 * imp_vel)
            # 原逻辑：if score > 2 then +20 (梯度断层)
            # 新逻辑：max(0, score - 2) * weight (连续梯度)
            # 含义：只要得分超过 2.0，每多一分，额外给予巨额奖励

            # 速度 Bonus：模拟原版 +20 的量级
            # 设定：当 r_base_vel 从 2.0 提升到 3.0 时，奖励增加约 10 分
            bonus_vel = 10.0 * max(0.0, r_base_vel - 2.0)

            # 2D Bonus：模拟原版 +10 的量级
            # 设定：门槛为 3.0
            bonus_2d = 5.0 * max(0.0, r_base_2d - 3.0)

            w_2d = 0.7
            w_3d = 0.3

            r_pos_total = w_2d * (r_base_2d + bonus_2d) + w_3d * r_base_3d

            # 组合速度项
            r_vel_total = r_base_vel + bonus_vel

            # 最终求和
            r_total = r_pos_total + r_vel_total

            # (可选) 调试打印，方便看 Bonus 是否触发
            if getattr(self, "debug_mode", False):
                print(
                    f"R_Total:{r_total:.2f} | 2D:{r_base_2d:.2f}+{bonus_2d:.2f} | Vel:{r_base_vel:.2f}+{bonus_vel:.2f}")

            return float(r_total)

        # # --- 推荐的 2D 奖励 (纯 2D 目标) ---
        # elif self.reward_setting in ['2dRMSEadv_kf_tanh', '2dRMSEratio_kf_tanh']:
        #     # 仅在需要时执行昂贵的坐标转换
        #     llh1 = coord.ecef2geodetic(gt_xyz)
        #     llh2 = coord.ecef2geodetic(kf_xyz)
        #     llh3 = coord.ecef2geodetic(rl_xyz)
        #     llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        #     llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        #     if self.reward_setting == '2dRMSEadv_kf_tanh':
        #         # 2D 优势 (KF - RL)，使用 tanh 缩放
        #         advantage_2d = llerr_kf - llerr_rl
        #         base_reward = np.tanh(advantage_2d * self.reward_scale)
        #     elif self.reward_setting == '2dRMSEratio_kf_tanh':
        #         # 2D 归一化比率，使用 tanh 压缩
        #         ratio_2d = (llerr_kf - llerr_rl) / (llerr_kf + 1e-6)
        #         base_reward = np.tanh(ratio_2d)
        # # --- 原始奖励设置 (保留对比) ---
        # else:
        #     # 在 reset 中更新 last_error, 否则第一次 step 会报错
        #     if self.last_error_rl is not None:
        #         self.last_error_rl = error_rl
        #     if self.last_llerr_rl is not None and llerr_rl is not None:
        #         self.last_llerr_rl = llerr_rl
        #     raise ValueError(f"Unknown reward setting: {self.reward_setting}")
        # # --- 3. 奖励塑形 (PBRS) (可选) ---
        # # 奖励智能体“相对于上一刻的自己”的进步
        # shaping_reward = 0.0
        # if self.enable_shaping:
        #     # 基于主要目标（3D 或 2D）来塑形
        #     if '2d' in self.reward_setting:
        #         # 使用 2D 误差进行塑形
        #         if llerr_rl is None:  # 确保 2D 误差已计算
        #             llh1 = coord.ecef2geodetic(gt_xyz)
        #             llh3 = coord.ecef2geodetic(rl_xyz)
        #             llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        #
        #         if self.last_llerr_rl is not None:
        #             shaping_reward = (self.last_llerr_rl - llerr_rl) * self.shaping_weight
        #         self.last_llerr_rl = llerr_rl  # 更新 2D 误差
        #         self.last_error_rl = error_rl  # 同时也更新 3D，以防万一
        #
        #     else:
        #         # 默认使用 3D 误差进行塑形
        #         if self.last_error_rl is not None:
        #             shaping_reward = (self.last_error_rl - error_rl) * self.shaping_weight
        #         self.last_error_rl = error_rl  # 更新 3D 误差
        #         # (如果需要，也可以更新 2D)
        # # --- 4. 最终奖励 ---
        # # 总奖励 = 基础奖励 (与 KF/OBS 对比) + 塑形奖励 (与自己对比)
        # return base_reward + shaping_reward


    def _print_perstep_info(self,timestep,obs_xyz,rl_xyz,gt_xyz,reward,rl_vel):
        print(
            f'{self.trajID_list[self.traj_idx]}, Time {timestep}/{self.timeend} '
            f'Baseline dist: [{np.abs(obs_xyz[0] - gt_xyz[0]): .2f}, {np.abs(obs_xyz[1] - gt_xyz[1]): .2f}, {np.abs(obs_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RL dist: [{np.abs(rl_xyz[0] - gt_xyz[0]): .2f}, {np.abs(rl_xyz[1] - gt_xyz[1]): .2f}, {np.abs(rl_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RMSEadv: {reward: 0.2e} m.'
            f'Vel:{rl_vel:.2f}m.')


    def _get_info(self, errors, traj_ID):
        return {
            'traj_idx': self.traj_idx,
            'traj_ID': traj_ID,
            'current_step': self.current_step,
            'baseline': self.baseline,
            'error': errors['rl'],
            'error_bl': errors['bl'],
            'error_kf': errors['kf'],
            'rl2gt_err': errors['rl2gt'],
            'rl_ll_err': errors['rl_ll'],
            'bl_ll_err': errors['bl_ll']
        }
    @staticmethod
    def _prepare_input(env_data,trajID_list,traj_idx):
        env_data['data_truth_dic'][trajID_list[traj_idx]] = env_data['data_truth_dic'][
            trajID_list[traj_idx]].copy().reset_index(drop=True)
        baseline = env_data['data_truth_dic'][trajID_list[traj_idx]].copy()
        losfeature = env_data['losfeature'][trajID_list[traj_idx]].copy()
        return baseline,losfeature

    def reset(self):
        self._select_traj()
        self.baseline,self.losfeature = self._prepare_input(self.env_data,self.trajID_list,self.traj_idx)
        self.current_step = self._initialize_current_step(self.baseline,self.traj_ratio)
        self._update_idx()
        self._load_trajdata_init_kf_velocity()
        obs = self._next_observation()
        self.total_reward = 0
        self.last_error_rl = None
        self.last_llerr_rl = None
        return obs

    def step(self, action):
        done= self._check_done()
        timestep = self.baseline.loc[self.cur_idx, 'UTCtime']
        predict_kg, delta_pos,delta_vel = self._parse_action(action)
        obs_xyz = self._get_position(self.baseline_model,self.baseline,self.cur_idx)
        kf_xyz = self._get_position('kf',self.baseline,self.cur_idx)
        gt_xyz = self._get_position('gt',self.baseline,self.cur_idx)
        pre_gt_xyz = self._get_position('gt',self.baseline,self.pre_idx)
        vel = self._get_velocity(self.baseline,self.cur_idx,self.selfvel)
        # velocity = np.sqrt(np.sum(vel_wls)**2)
        x_rl = obs_xyz + delta_pos
        vel_rl = vel + delta_vel
        rl_pos,rl_vel,dt = self.RL4KFGSDC(x_rl,vel_rl,predict_kg,policy=True)
        self._update_baseline(rl_pos,rl_vel)
        feature_dict = self._get_features(rl_pos, rl_vel)
        self._update_data_truth_dic(feature_dict)
        reward = self._compute_reward(rl_pos,gt_xyz,obs_xyz,kf_xyz,pre_gt_xyz,dt,vel,rl_vel)
        self.total_reward += reward
        if self.step_print:
            self._print_perstep_info(timestep, obs_xyz, rl_pos, gt_xyz, reward,rl_vel)
        self.current_step += 1
        self._update_idx()
        obs = [] if done else self._next_observation()
        llh_gt =coord.ecef2geodetic(gt_xyz)
        llh_obs =coord.ecef2geodetic(obs_xyz)
        llh_rl =coord.ecef2geodetic(rl_pos)
        llerr_obs = haversine((llh_gt[0], llh_gt[1]), (llh_obs[0], llh_obs[1]), unit='m')
        llerr_rl = haversine((llh_gt[0], llh_gt[1]), (llh_rl[0], llh_rl[1]), unit='m')
        self.rollout_rl_error_sum += llerr_rl
        self.rollout_bl_error_sum += llerr_obs
        self.rollout_step_count += 1
        errors = {
            'rl': np.linalg.norm(rl_pos - gt_xyz),
            'bl': np.linalg.norm(obs_xyz - gt_xyz),
            'kf': np.linalg.norm(kf_xyz - gt_xyz),
            'rl2gt': np.sum((rl_pos - gt_xyz) ** 2),
            'rl_ll': llerr_rl,
            'bl_ll': llerr_obs
        }
        info = self._get_info(errors, self.trajID_list[self.traj_idx])
        # 防止除以0 (虽然理论上不可能，加个 max 安全一点)
        steps = max(1, self.rollout_step_count)

        # 1. 计算本回合的平均误差
        avg_rl_error = self.rollout_rl_error_sum / steps
        avg_bl_error = self.rollout_bl_error_sum / steps

        # 2. 将平均值写入 info 字典
        # 这些 key (episode_mean_rl_error) 就是你在 Callback 里要读取的名字
        info['episode_mean_rl_error'] = avg_rl_error
        info['episode_mean_bl_error'] = avg_bl_error

        # (可选) 如果你想在 log 里看到这一回合走了多少步
        info['episode_steps'] = steps
        return obs,reward,done,info


class Kt_Withvel_dt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, env_data, cfg, traj_type, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, kg_action_scale, conv_corr, allcorrect,
                 selfvel, traj_ratio, tripid_range_ratio, step_print, Q_static):
        super(Kt_Withvel_dt, self).__init__()
        self.rollout_rl_error_sum = 0
        self.rollout_step_count = 0
        self.rollout_bl_error_sum = 0
        self.state = None
        self.step_print = step_print
        self.sigma_mahalanobis = sigma_mahalanobis
        self.outlier_velocity = outlier_velocity
        self.max_visible_sat = 24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        self.cfg = cfg
        self.conv_corr = conv_corr
        self.selfvel = selfvel
        self.allcorrect = allcorrect
        self.env_data = env_data
        self.trajID_list = env_data[traj_type]
        self.traj_type = traj_type
        self.traj_ratio = [traj_ratio[0], traj_ratio[1] * tripid_range_ratio]
        self.trajdata_range = trajdata_range
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
             'pos': spaces.Box(low=-1, high=1, shape=(1, (self.pos_num - 1) * 3), dtype=np.float),
             'vel': spaces.Box(low=-1, high=1, shape=(1, 3 * (self.pos_num - 1)), dtype=np.float)})

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.kg_diagonal = kg_action_scale['diagonal']
        self.kg_up = kg_action_scale['up']
        self.kg_down = kg_action_scale['down']
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1],
                                       shape=(1, 12 + 3 + 3),
                                       dtype=np.float)
        self.total_reward = 0
        self.baseline_model = baseline_mod
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        if self.trajdata_sort == 'sorted':
            self.traj_idx = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])

        self.data_truth_dic = self.env_data['data_truth_dic'].copy()
        self.verbose = 2
        self.Q_static = Q_static
        # =========new reward
        self.enable_shaping = False  # bool 标志：是否启用奖励塑形
        self.shaping_weight = 0.1  # 塑形奖励的权重
        self.last_error_rl = None  # 存储上一步的误差 (3D)
        self.last_llerr_rl = None  # 存储上一步的误差 (2D)
        # 2. 奖励缩放
        self.reward_scale = 0.2

    def reset_rollout_metrics(self):
        """
        这个函数不是给 gym 用的，是给 Callback 在 rollout 开始时强制调用的
        """
        self.rollout_rl_error_sum = 0.0
        self.rollout_bl_error_sum = 0.0
        self.rollout_step_count = 0

    def _select_traj(self):
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.traj_idx = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.traj_idx = self.traj_idx + 1
            if self.traj_idx > self.trajdata_range[1]:
                self.traj_idx = self.trajdata_range[0]

    def _load_trajdata_init_kf_velocity(self):
        if 'spp' in self.baseline_model:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_model == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']
        self.baseline['VX_RLpredict'] = self.baseline['VXEcefMeters_wls']
        self.baseline['VY_RLpredict'] = self.baseline['VYEcefMeters_wls']
        self.baseline['VZ_RLpredict'] = self.baseline['VZEcefMeters_wls']

        if self.traj_ratio[0] > 0:  # 只要剩下部分轨迹的定位结果
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['X_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Y_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['Z_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VX_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VY_RLpredict']] = None
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[0:self.current_step - 1,
            ['VZ_RLpredict']] = None

            self.baseline.loc[0:self.current_step - 1, ['X_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['Z_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VX_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VY_RLpredict']] = None
            self.baseline.loc[0:self.current_step - 1, ['VZ_RLpredict']] = None

        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])
        # set parameter for KF
        p0 = initial_P
        v0 = initial_V
        variances = [p0 ** 2] * 3 + [v0 ** 2] * 3
        self.P = np.diag(variances)  # 协方差矩阵
        self.sigma_m_pos = sigma_m_pos
        self.sigma_m_vel = sigma_m_vel
        variances_QR = [self.sigma_m_pos ** 2] * 3 + [self.sigma_m_vel ** 2] * 3
        self.Q_initial = np.diag(variances_QR)
        self.R = np.diag(variances_QR)  # Measurement noise
        self.datatime = self.baseline['UTCtime']  # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values) - 1, 'UTCtime']
        self.sigma_a = sigma_a

        obs_pos = self._get_position(self.baseline_model, self.baseline, self.pre_idx)
        wls_vel = self._get_velocity(self.baseline, self.pre_idx, self.selfvel)
        self.state = np.hstack([obs_pos, wls_vel])

    def _normalize_pos(self, state):
        # state[0] = (state[0] - xecef_min) / (xecef_max-xecef_min)
        # state[1] = (state[1] - yecef_min) / (yecef_max-yecef_min)
        # state[2] = (state[2] - zecef_min) / (zecef_max-zecef_min)
        # state[0] = (state[0] - np.mean(state[0])) / np.std(state[0])
        # state[1] = (state[1] - np.mean(state[1])) / np.std(state[1])
        # state[2] = (state[2] - np.mean(state[2])) / np.std(state[2])
        state[0] = state[0] / 20
        state[1] = state[1] / 20
        state[2] = state[2] / 20
        return state

    def _normalize_vel(self, state):
        state[0] = state[0] / outlier_velocity
        state[1] = state[1] / outlier_velocity
        state[2] = state[2] / outlier_velocity
        return state

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:, 1] = (gnss[:, 1]) / res_max
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _obs_padding(self, obs):
        # --- 填充数组以确保尺寸正确 ---
        expected_cols = self.pos_num - 1  # 期望的列数为4
        current_cols = obs.shape[1]  # 当前数组的列数
        # 如果列数不足，则进行填充
        if current_cols < expected_cols:
            # 计算需要填充的列数
            padding_cols = expected_cols - current_cols
            # 创建一个需要填充的零数组
            padding = np.zeros((3, padding_cols))
            # 将原始数组和零数组拼接在一起
            obs = np.concatenate((obs, padding), axis=1)
        # --- 填充结束 ---
        return obs

    def _process_gnss_feature(self):
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp = self.losfeature[self.datatime[self.cur_idx]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([self.max_visible_sat, self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5]  # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat, :]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp
        return obs_feature

    def _next_observation(self):

        obs = np.array([
            self.baseline.loc[self.current_step: self.pre_idx, 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'Z_RLpredict'].values])
        vel = np.array([
            self.baseline.loc[self.current_step: self.pre_idx, 'VX_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'VY_RLpredict'].values,
            self.baseline.loc[self.current_step: self.pre_idx, 'VZ_RLpredict'].values])
        if 'spp' in self.baseline_model:
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_spp']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_spp']]],
                            axis=1)
            vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_wls']],
                                  [self.baseline.loc[self.cur_idx, 'VYEcefMeters_wls']],
                                  [self.baseline.loc[self.cur_idx, 'VZEcefMeters_wls']]],
                            axis=1)
        elif self.baseline_model == 'rtk':
            obs = np.append(obs, [[self.baseline.loc[self.cur_idx, 'XEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'YEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'ZEcefMeters_rtk']]],
                            axis=1)
            vel = np.append(vel, [[self.baseline.loc[self.cur_idx, 'VXEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'VYEcefMeters_rtk']],
                                  [self.baseline.loc[self.cur_idx, 'VZEcefMeters_rtk']]],
                            axis=1)
        obs = np.diff(obs, n=1, axis=-1)  # 位置差分
        vel = np.diff(vel, n=1, axis=-1)
        obs = self._obs_padding(obs)
        vel = self._obs_padding(vel)
        obs = self._normalize_pos(obs)
        vel = self._normalize_vel(vel)
        obs_feature = self._process_gnss_feature()
        obs_all = {'pos': obs.reshape(1, 3 * (self.pos_num - 1), order='F'),
                   'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'vel': vel.reshape(1, 3 * (self.pos_num - 1), order='F')}
        self.obs = obs_all
        return obs_all

    def _update_idx(self):
        self.max_index = len(self.baseline) - 1
        self.cur_idx = int(min(self.current_step + (self.pos_num - 1), self.max_index))
        self.pre_idx = int(min(self.current_step + (self.pos_num - 2), self.max_index))

    @staticmethod
    def _initialize_current_step(baseline, traj_ratio):
        return int(np.ceil(len(baseline) * traj_ratio[0]))

    def _check_done(self):
        return self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values) * self.traj_ratio[-1] \
            - self.pos_num - outlayer_in_end_ecef

    def _parse_action(self, action):
        action = action.reshape(1, -1)

        # fill_indices = np.array([0, 3, 7, 10, 14, 17, 18, 21, 25, 28, 32, 35])
        diag_indices_set = {0, 7, 14, 21, 28, 35}
        up_indices_set = {3, 10, 17}
        down_indices_set = {18, 25, 32}

        fill_indices = np.sort(list(diag_indices_set | up_indices_set | down_indices_set))

        # 2. 根据索引所属的集合选择对应的 scale
        current_scales = np.array([
            self.kg_diagonal if idx in diag_indices_set
            else self.kg_up if idx in up_indices_set
            else self.kg_down
            for idx in fill_indices
        ])

        matrix_values = action[0, :12]
        scaled_values = matrix_values * current_scales

        predict_kg_flat = np.zeros(36)
        predict_kg_flat[fill_indices] = scaled_values
        predict_kg = predict_kg_flat.reshape(6, 6)

        delta_pos = action[0, 12:15] * self.continuous_action_scale
        delta_vel = action[0, 15:18] * self.continuous_action_scale / 2
        # delta_vel = action[0, 15:18] * 0

        return predict_kg, delta_pos, delta_vel

    @staticmethod
    def _get_position(source, baseline, cur_idx):
        keyword = {
            'gt': {
                'X': 'ecefX_gt',
                'Y': 'ecefY_gt',
                'Z': 'ecefZ_gt'
            },
            'kf': {
                'X': 'XEcefMeters_kf',
                'Y': 'YEcefMeters_kf',
                'Z': 'ZEcefMeters_kf'
            },
            'spp': {
                'X': 'XEcefMeters_spp',
                'Y': 'YEcefMeters_spp',
                'Z': 'ZEcefMeters_spp'
            },
            'rtk': {
                'X': 'XEcefMeters_rtk',
                'Y': 'YEcefMeters_rtk',
                'Z': 'ZEcefMeters_rtk'
            }
        }[source]

        row = baseline.iloc[cur_idx]  # <— iloc
        return np.array([row[keyword['X']], row[keyword['Y']], row[keyword['Z']]])

    @staticmethod
    def _get_velocity(baseline, cur_idx, _):
        row = baseline.iloc[cur_idx]  # <— iloc
        vel = np.array([row['VXEcefMeters_wls'], row['VYEcefMeters_wls'], row['VZEcefMeters_wls']])
        return vel

    # def _get_xyz(self,method):
    #     if method == 'gt':
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'ecefX_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ecefY_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ecefZ_{method}']
    #         ])
    #     else:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx,f'XEcefMeters_{method}'],
    #             self.baseline.loc[self.cur_idx, f'YEcefMeters_{method}'],
    #             self.baseline.loc[self.cur_idx, f'ZEcefMeters_{method}']
    #         ])
    # def _get_velocity(self):
    #     if self.selfvel:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'VXEcefMeters_self'],
    #             self.baseline.loc[self.cur_idx, f'VYEcefMeters_self'],
    #             self.baseline.loc[self.cur_idx, f'VZEcefMeters_self']
    #         ])
    #     else:
    #         return np.array([
    #             self.baseline.loc[self.cur_idx, f'VXEcefMeters_wls'],
    #             self.baseline.loc[self.cur_idx, f'VYEcefMeters_wls'],
    #             self.baseline.loc[self.cur_idx, f'VZEcefMeters_wls']
    #         ])
    def _check_velocity_outlier(self, velocity):
        if np.sqrt(np.sum(velocity ** 2)) > outlier_velocity:
            return True
        else:
            return False

    # def _kf_step_(self,state,vel,measurement,predict_kg,Q,R,utc_pre,utc_cur):
    #     F = np.eye(3)
    #     H = np.eye(3)
    #     x_predict = F @ state + vel.T *(utc_cur-utc_pre)
    #     P_predict = F @ self.P @ F.T + Q
    #     d = dist_err_XYZ(measurement,H @ x_predict)
    #     if d < self.sigma_mahalanobis:
    #         y = measurement.T - H @ x_predict
    #         S = H @ P_predict @ H.T + R
    #         if np.all(self.Kalman_gain == 0) :
    #             K = P_predict @ H.T @ np.linalg.inv(S)
    #             K += predict_kg
    #             self.Kalman_gain = K
    #         else:
    #             K = self.Kalman_gain + predict_kg
    #         x_update = x_predict + K @ y
    #         self.P = (np.eye(3) - K @ H) @ P_predict
    #         self.covx = R
    #     else:
    #         self.P = P_predict + 100 * Q
    #         x_update = x_predict
    #     return x_update



    def _deal_with_dt_outline(self, dt, pos_rl, vel_rl, utc_cur, utc_pre):
        # 处理时间异常点，时间中断：位置对齐观测，增大不确定性
        if dt > interrupt_time or np.isnan(vel_rl).any():  # 增大噪声，用上一步预测速度替换当前速度
            # self.P += 100.0 * np.block([
            #     [(self.sigma_m_pos ** 2) * np.eye(3), np.zeros((3, 3))],
            #     [np.zeros((3, 3)), self.sigma_m_vel ** 2 * np.eye(3)]
            # ])
            vel_rl = self.state[3:]
            if dt > interrupt_time:
                if self.verbose >= 2:
                    logging.warning(f'time jump {dt}s, utc={utc_cur}, utc_pre={utc_pre},'
                                    f'vel_cur={self._get_velocity(self.baseline, self.cur_idx, self.selfvel)},vel_pre={self._get_velocity(self.baseline, self.pre_idx, self.selfvel)}')
            elif np.isnan(vel_rl).any():
                if self.verbose >= 2:
                    logging.warning(f'velocity Nan')
            # self.state[:3] = pos_rl

        return pos_rl, vel_rl

    def _deal_with_vel_outline(self, vel_rl, utc_cur, utc_pre, Q):
        # 处理速度异常点
        vel_norm = np.linalg.norm(vel_rl)  # 提取出来，避免多次计算
        if vel_norm > self.outlier_velocity:
            if self.verbose >= 2:
                logging.warning(
                    f'velocity jump {vel_norm: .2f}m/s, utc={utc_cur}, utc_pre={utc_pre}')

            # 执行平滑/修正
            vel_rl = (self.state[3:] + vel_rl) / 2

            # [新增] 打印修正后的速度
            if self.verbose >= 2:
                logging.warning(f'-> corrected velocity: {np.linalg.norm(vel_rl): .2f}m/s')

            # 重置协方差 P
            self.P = np.block([
                [(self.sigma_m_pos ** 2) * 100 * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), (self.sigma_m_vel ** 2) * 100 * np.eye(3)]
            ])
        return vel_rl

    def RL4KFGSDC(self, pos_rl, vel_rl, predict_kg, policy=True):
        idx_cur = self.current_step + (self.pos_num - 1)
        idx_pre = self.current_step + (self.pos_num - 2)
        utc_cur = self.datatime[idx_cur]
        utc_pre = self.datatime[idx_pre]
        dt = float(max(utc_cur - utc_pre, 1e-3))
        if self.Q_static:
            Q = self.Q_initial
        else:
            # 预计算标量，减少重复乘法
            sigma2 = self.sigma_a ** 2
            q11 = (dt ** 4) / 4.0 * sigma2
            q12 = (dt ** 3) / 2.0 * sigma2
            q22 = (dt ** 2) * sigma2
            I3 = np.eye(3)
            Q = np.block([
                [q11 * I3, q12 * I3],
                [q12 * I3, q22 * I3],
            ])

        R = self.R
        F = np.eye(6, dtype=float)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        # H = np.eye(6, dtype=float)
        interrupt_time = getattr(self, 'interrupt_time', 5.0)  # 防止变量未定义
        is_interrupted = dt > interrupt_time
        is_vel_nan = np.isnan(vel_rl).any()
        if is_interrupted or is_vel_nan:
            # [关键修正] 不仅要重置 state，还要重置 P (协方差)，否则 Filter 会过度自信
            if is_interrupted:
                if self.verbose >= 2:
                    logging.warning(f'Time jump {dt:.2f}s, Resetting Filter...')
            elif is_vel_nan:
                if self.verbose >= 2:
                    logging.warning(f'Velocity NaN, Resetting Filter...')

            # 强制重置状态
            # 如果 vel_rl 是 NaN，最好填 0 或者上一时刻速度，防止污染 state
            safe_vel = vel_rl if not is_vel_nan else np.zeros(3)
            self.state = np.hstack([pos_rl, safe_vel])

            # [新增] 重置 P 矩阵，给予较大的初始不确定度
            self.P = np.block([
                [(self.sigma_m_pos ** 2) * 100 * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), (self.sigma_m_vel ** 2) * 100 * np.eye(3)]
            ])
            # 统一返回值格式
            return pos_rl, safe_vel, dt
        else:
            vel_rl = self._deal_with_vel_outline(vel_rl, utc_cur, utc_pre, Q)
            self._kf_step(pos_rl, vel_rl, F, predict_kg, Q, R, utc_pre, utc_cur)
            self.v_pre = self.state[3:].copy()
            rl_pos = self.state[:3].copy()
            rl_vel = self.state[3:].copy()
            if not policy:
                spp_pos = self.baseline.loc[idx_cur, ['XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']].values
                self.state[:3] = spp_pos
                rl_pos = spp_pos

            return rl_pos, rl_vel, dt

    def _kf_step(self, pos_rl, vel_rl, F, predict_kg, Q, R, utc_pre, utc_cur):
        # 1. Prediction Step
        z = np.hstack([pos_rl, vel_rl])
        x_predict = F @ self.state
        P_predict = F @ self.P @ F.T + Q
        y = z.T - x_predict
        S = P_predict + R
        # whiten = np.inf
        # 2. Measurement Update Check (Mahalanobis Gate)
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = y.T @ S_inv @ y
            whiten = float(np.sqrt(max(0, mahalanobis_sq)))
        except np.linalg.LinAlgError:
            if self.verbose >= 2:
                logging.warning(f'Matrix inversion failed (Singular S), utc={utc_cur}')
            whiten = np.inf

        if np.isfinite(whiten) and (whiten <= self.sigma_mahalanobis):
            K = P_predict @ S_inv  # (6,6)
            K_actual = K + predict_kg
            self.state = x_predict + K_actual @ y
            self.P = (np.eye(6) - K) @ P_predict
        else:
            if self.verbose >= 2:
                d = np.linalg.norm(z[:3] - x_predict[:3])
                logging.warning(f'distance {d:.2f} {whiten:.2f}')
            self.state = x_predict
            inflation = np.block([
                [(self.sigma_m_pos ** 2) * 100 * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), (self.sigma_m_vel ** 2) * 100 * np.eye(3)]
            ])
            self.P = P_predict + inflation
    def _update_baseline(self, rl_xyz, rl_vel):
        pos_map = {
            'X': 0,
            'Y': 1,
            'Z': 2,
        }
        for key, value in pos_map.items():
            self.baseline.loc[self.cur_idx, [f'{key}_RLpredict']] = rl_xyz[value]
            self.baseline.loc[self.cur_idx, [f'V{key}_RLpredict']] = rl_vel[value]

    def _get_features(self, rl_pos, rl_vel):
        rl_pos = np.asarray(rl_pos)
        feature_tmp = self.losfeature[self.datatime[self.cur_idx]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        # Velocity = np.linalg.norm(rl_vel)
        feature_dict = {
            'X_RLpredict': rl_pos[0],
            'Y_RLpredict': rl_pos[1],
            'Z_RLpredict': rl_pos[2],
            'VX_RLpredict': rl_vel[0],
            'VY_RLpredict': rl_vel[1],
            'VZ_RLpredict': rl_vel[2],
            'CN0_mean': CN0_mean,
            'EA_mean': EA_mean,
            'PR_mean': PR_mean,
            'satnum': satnum,
            # 'velocity': Velocity
        }
        return feature_dict

    def _update_data_truth_dic(self, feature_dict):
        for key, value in feature_dict.items():
            self.env_data['data_truth_dic'][self.trajID_list[self.traj_idx]].loc[self.cur_idx, [key]] = value

    def _compute_reward_(self, rl_xyz, gt_xyz, obs_xyz, kf_xyz):
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_obs = np.linalg.norm(obs_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        llh1 = coord.ecef2geodetic(gt_xyz)
        llh2 = coord.ecef2geodetic(kf_xyz)
        llh3 = coord.ecef2geodetic(rl_xyz)
        llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        if self.reward_setting == 'RMSE':
            return 10 - error_rl
        elif self.reward_setting == 'RMSEadv':
            return error_obs - error_rl
        elif self.reward_setting == 'RMSEadv_kf':
            return error_kf - error_rl
        elif self.reward_setting == 'RMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6)
        elif self.reward_setting == '2dRMSEratio_kf':
            return 10 * (error_kf - error_rl) / (error_kf + 1e-6) + llerr_kf - llerr_rl
        else:
            raise ValueError(f"Unknown reward setting: {self.reward_setting}")

    def _compute_reward(self, rl_xyz, gt_xyz, obs_xyz, kf_xyz , pre_gt_xyz , dt,vel,rl_vel):

        # --- 1. 基础 3D 误差计算 (常用) ---
        error_rl = np.linalg.norm(rl_xyz - gt_xyz)
        error_kf = np.linalg.norm(kf_xyz - gt_xyz)
        base_reward = 0.0
        llerr_rl = None  # 默认为空，仅在 2D 模式下计算
        # --- 2. 计算基础奖励 (根据设置) ---
        if self.reward_setting == 'RMSEadv_kf_tanh':
            # 3D 优势 (KF - RL)，使用 tanh 缩放到 [-1, 1]
            advantage = error_kf - error_rl
            base_reward = np.tanh(advantage * self.reward_scale)
        elif self.reward_setting == 'RMSEratio_kf_tanh':
            # 3D 归一化比率，使用 tanh 压缩到 [-1, 1]
            ratio = (error_kf - error_rl) / (error_kf + 1e-6)
            base_reward = np.tanh(ratio)

        elif self.reward_setting == 'RMSEmixratio_kf_tanh':

            # [位置 2D] (保留你原本的 Haversine 逻辑，准确度最高)
            llh_gt = coord.ecef2geodetic(gt_xyz)
            llh_kf = coord.ecef2geodetic(kf_xyz)
            llh_rl = coord.ecef2geodetic(rl_xyz)
            vel_gt = (gt_xyz - pre_gt_xyz) / dt
            err_2d_kf = haversine((llh_gt[0], llh_gt[1]), (llh_kf[0], llh_kf[1]), unit='m')
            err_2d_rl = haversine((llh_gt[0], llh_gt[1]), (llh_rl[0], llh_rl[1]), unit='m')
            # [速度] (假设你已保证 rl_vel, vel, vel_gt 坐标系一致且非空)
            # vel 代表基准算法(WLS/KF)的速度
            err_vel_base = np.linalg.norm(vel - vel_gt)
            err_vel_rl = np.linalg.norm(rl_vel - vel_gt)
            eps = 1e-6
            imp_2d = (err_2d_kf - err_2d_rl) / (err_2d_kf + eps)
            imp_3d = (error_kf - error_rl) / (error_kf + eps)
            imp_vel = (err_vel_base - err_vel_rl) / (err_vel_base + eps)

            r_base_2d = 10.0 * np.tanh(3.0 * imp_2d)
            r_base_3d = 10.0 * np.tanh(3.0 * imp_3d)
            r_base_vel = 10.0 * np.tanh(1.0 * imp_vel)
            # 原逻辑：if score > 2 then +20 (梯度断层)
            # 新逻辑：max(0, score - 2) * weight (连续梯度)
            # 含义：只要得分超过 2.0，每多一分，额外给予巨额奖励

            # 速度 Bonus：模拟原版 +20 的量级
            # 设定：当 r_base_vel 从 2.0 提升到 3.0 时，奖励增加约 10 分
            bonus_vel = 10.0 * max(0.0, r_base_vel - 2.0)

            # 2D Bonus：模拟原版 +10 的量级
            # 设定：门槛为 3.0
            bonus_2d = 5.0 * max(0.0, r_base_2d - 3.0)

            w_2d = 0.7
            w_3d = 0.3

            r_pos_total = w_2d * (r_base_2d + bonus_2d) + w_3d * r_base_3d

            # 组合速度项
            r_vel_total = r_base_vel + bonus_vel

            # 最终求和
            r_total = r_pos_total + r_vel_total

            # (可选) 调试打印，方便看 Bonus 是否触发
            if getattr(self, "debug_mode", False):
                print(
                    f"R_Total:{r_total:.2f} | 2D:{r_base_2d:.2f}+{bonus_2d:.2f} | Vel:{r_base_vel:.2f}+{bonus_vel:.2f}")

            return float(r_total)
        # # --- 推荐的 2D 奖励 (纯 2D 目标) ---
        # elif self.reward_setting in ['2dRMSEadv_kf_tanh', '2dRMSEratio_kf_tanh']:
        #     # 仅在需要时执行昂贵的坐标转换
        #     llh1 = coord.ecef2geodetic(gt_xyz)
        #     llh2 = coord.ecef2geodetic(kf_xyz)
        #     llh3 = coord.ecef2geodetic(rl_xyz)
        #     llerr_kf = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        #     llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        #     if self.reward_setting == '2dRMSEadv_kf_tanh':
        #         # 2D 优势 (KF - RL)，使用 tanh 缩放
        #         advantage_2d = llerr_kf - llerr_rl
        #         base_reward = np.tanh(advantage_2d * self.reward_scale)
        #     elif self.reward_setting == '2dRMSEratio_kf_tanh':
        #         # 2D 归一化比率，使用 tanh 压缩
        #         ratio_2d = (llerr_kf - llerr_rl) / (llerr_kf + 1e-6)
        #         base_reward = np.tanh(ratio_2d)
        # # --- 原始奖励设置 (保留对比) ---
        # else:
        #     # 在 reset 中更新 last_error, 否则第一次 step 会报错
        #     if self.last_error_rl is not None:
        #         self.last_error_rl = error_rl
        #     if self.last_llerr_rl is not None and llerr_rl is not None:
        #         self.last_llerr_rl = llerr_rl
        #     raise ValueError(f"Unknown reward setting: {self.reward_setting}")
        # # --- 3. 奖励塑形 (PBRS) (可选) ---
        # # 奖励智能体“相对于上一刻的自己”的进步
        # shaping_reward = 0.0
        # if self.enable_shaping:
        #     # 基于主要目标（3D 或 2D）来塑形
        #     if '2d' in self.reward_setting:
        #         # 使用 2D 误差进行塑形
        #         if llerr_rl is None:  # 确保 2D 误差已计算
        #             llh1 = coord.ecef2geodetic(gt_xyz)
        #             llh3 = coord.ecef2geodetic(rl_xyz)
        #             llerr_rl = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
        #
        #         if self.last_llerr_rl is not None:
        #             shaping_reward = (self.last_llerr_rl - llerr_rl) * self.shaping_weight
        #         self.last_llerr_rl = llerr_rl  # 更新 2D 误差
        #         self.last_error_rl = error_rl  # 同时也更新 3D，以防万一
        #
        #     else:
        #         # 默认使用 3D 误差进行塑形
        #         if self.last_error_rl is not None:
        #             shaping_reward = (self.last_error_rl - error_rl) * self.shaping_weight
        #         self.last_error_rl = error_rl  # 更新 3D 误差
        #         # (如果需要，也可以更新 2D)
        # # --- 4. 最终奖励 ---
        # # 总奖励 = 基础奖励 (与 KF/OBS 对比) + 塑形奖励 (与自己对比)
        # return base_reward + shaping_reward

    def _print_perstep_info(self, timestep, obs_xyz, rl_xyz, gt_xyz, reward, rl_vel):
        print(
            f'{self.trajID_list[self.traj_idx]}, Time {timestep}/{self.timeend} '
            f'Baseline dist: [{np.abs(obs_xyz[0] - gt_xyz[0]): .2f}, {np.abs(obs_xyz[1] - gt_xyz[1]): .2f}, {np.abs(obs_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RL dist: [{np.abs(rl_xyz[0] - gt_xyz[0]): .2f}, {np.abs(rl_xyz[1] - gt_xyz[1]): .2f}, {np.abs(rl_xyz[2] - gt_xyz[2]): .2f}] m, '
            f'RMSEadv: {reward: 0.2e} m.'
            f'Vel:{rl_vel:.2f}m.')

    def _get_info(self, errors, traj_ID):
        return {
            'traj_idx': self.traj_idx,
            'traj_ID': traj_ID,
            'current_step': self.current_step,
            'baseline': self.baseline,
            'error': errors['rl'],
            'error_bl': errors['bl'],
            'error_kf': errors['kf'],
            'rl2gt_err': errors['rl2gt'],
            'rl_ll_err': errors['rl_ll'],
            'bl_ll_err': errors['bl_ll']
        }

    @staticmethod
    def _prepare_input(env_data, trajID_list, traj_idx):
        env_data['data_truth_dic'][trajID_list[traj_idx]] = env_data['data_truth_dic'][
            trajID_list[traj_idx]].copy().reset_index(drop=True)
        baseline = env_data['data_truth_dic'][trajID_list[traj_idx]].copy()
        losfeature = env_data['losfeature'][trajID_list[traj_idx]].copy()
        return baseline, losfeature

    def reset(self):
        self._select_traj()
        self.baseline, self.losfeature = self._prepare_input(self.env_data, self.trajID_list, self.traj_idx)
        self.current_step = self._initialize_current_step(self.baseline, self.traj_ratio)
        self._update_idx()
        self._load_trajdata_init_kf_velocity()
        obs = self._next_observation()
        self.total_reward = 0
        self.last_error_rl = None
        self.last_llerr_rl = None
        return obs

    def step(self, action):
        done = self._check_done()
        timestep = self.baseline.loc[self.cur_idx, 'UTCtime']
        predict_kg, delta_pos, delta_vel = self._parse_action(action)
        obs_xyz = self._get_position(self.baseline_model, self.baseline, self.cur_idx)
        kf_xyz = self._get_position('kf', self.baseline, self.cur_idx)
        gt_xyz = self._get_position('gt', self.baseline, self.cur_idx)
        pre_gt_xyz = self._get_position('gt', self.baseline, self.pre_idx)
        vel = self._get_velocity(self.baseline, self.cur_idx, self.selfvel)
        # velocity = np.sqrt(np.sum(vel_wls)**2)
        x_rl = obs_xyz + delta_pos
        vel_rl = vel + delta_vel
        rl_pos, rl_vel,dt = self.RL4KFGSDC(x_rl, vel_rl, predict_kg, policy=True)
        self._update_baseline(rl_pos, rl_vel)
        feature_dict = self._get_features(rl_pos, rl_vel)
        self._update_data_truth_dic(feature_dict)
        reward = self._compute_reward(rl_pos,gt_xyz,obs_xyz,kf_xyz,pre_gt_xyz,dt,vel,rl_vel)
        self.total_reward += reward
        if self.step_print:
            self._print_perstep_info(timestep, obs_xyz, rl_pos, gt_xyz, reward, rl_vel)
        self.current_step += 1
        self._update_idx()
        obs = [] if done else self._next_observation()
        llh_gt = coord.ecef2geodetic(gt_xyz)
        llh_obs = coord.ecef2geodetic(obs_xyz)
        llh_rl = coord.ecef2geodetic(rl_pos)
        llerr_obs = haversine((llh_gt[0], llh_gt[1]), (llh_obs[0], llh_obs[1]), unit='m')
        llerr_rl = haversine((llh_gt[0], llh_gt[1]), (llh_rl[0], llh_rl[1]), unit='m')
        self.rollout_rl_error_sum += llerr_rl
        self.rollout_bl_error_sum += llerr_obs
        self.rollout_step_count += 1
        errors = {
            'rl': np.linalg.norm(rl_pos - gt_xyz),
            'bl': np.linalg.norm(obs_xyz - gt_xyz),
            'kf': np.linalg.norm(kf_xyz - gt_xyz),
            'rl2gt': np.sum((rl_pos - gt_xyz) ** 2),
            'rl_ll': llerr_rl,
            'bl_ll': llerr_obs
        }
        info = self._get_info(errors, self.trajID_list[self.traj_idx])
        # 防止除以0 (虽然理论上不可能，加个 max 安全一点)
        steps = max(1, self.rollout_step_count)

        # 1. 计算本回合的平均误差
        avg_rl_error = self.rollout_rl_error_sum / steps
        avg_bl_error = self.rollout_bl_error_sum / steps

        # 2. 将平均值写入 info 字典
        # 这些 key (episode_mean_rl_error) 就是你在 Callback 里要读取的名字
        info['episode_mean_rl_error'] = avg_rl_error
        info['episode_mean_bl_error'] = avg_bl_error

        # (可选) 如果你想在 log 里看到这一回合走了多少步
        info['episode_steps'] = steps
        return obs, reward, done, info