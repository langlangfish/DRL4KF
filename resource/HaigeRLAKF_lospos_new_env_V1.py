#强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
from env.env_param_new import *
from scipy.spatial import distance
import sys
sys.path.append("../src")
from funcs.utilis import *
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error
import simdkalman
step_print=False
#导入数据
dir_path = '/mnt/sdb/home/tangjh/DYdata2024/' # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
gnss_trig = True
new_version = True # 只使用0620后新版本V1.0.0的数据
selfvel = True
# sigma_v = 1.2 # 3.1
"""
losfeature: 0:satid,1:伪距残差,2-4:视距矢量,5：CNR,6:高度角，7：方位角
数据处理V1: 中断分割时间设为20s，
数据处理V2: 中断分割时间设为50s，剔除异常值更多（目前使用V2）
"""

# load raw data for position data
if selfvel:
    with open(dir_path+'env/raw_baseline_Haige_new_interupt_selfvel_V2.pkl', "rb") as file:
        data_truth_dic = pickle.load(file)
    file.close()
# load gnss features
with open(dir_path + 'env/processed_features_BDS_Haige_new_interupt_selfvel_V2.pkl', "rb") as file:
    losfeature = pickle.load(file)
file.close()

if new_version:
    traj_sum_df = pd.read_csv(f'{dir_path}/env/raw_tripID_BDS_Haige_new_interupt_selfvel_V2.csv') # _after0620
else:
    traj_sum_df = pd.read_csv(f'{dir_path}/env/raw_tripID_BDS_Haige_new_interupt_V2.csv')  # load trajectory ID and type
# construct trajID list
tripID_to_remove = ["20250703-PM_2-forest1-0", "20250703-PM_1-openroad3-1",'20250703-PM_2-openroad3-1'] # 剔除异常数据的id
traj_sum_df = traj_sum_df[~traj_sum_df['tripID'].isin(tripID_to_remove)]
traj_openroad = traj_sum_df.loc[(traj_sum_df['Type']=='openroad')]['tripID'].values.tolist()
traj_canyon = traj_sum_df.loc[(traj_sum_df['Type']=='canyon')]['tripID'].values.tolist()
traj_highway = traj_sum_df.loc[(traj_sum_df['Type']=='highway')]['tripID'].values.tolist()
traj_forest = traj_sum_df.loc[(traj_sum_df['Type']=='forest')]['tripID'].values.tolist()
traj_overpass = traj_sum_df.loc[(traj_sum_df['Type']=='overpass')]['tripID'].values.tolist()
traj_full = traj_sum_df['tripID'].values.tolist()
# traj_full_nocanyon = [x for x in traj_full if 'canyon' not in x] # 不用canyon作为训练数据
set_exclude = set(exclude_canyon)
traj_full_nocanyon = [x for x in traj_full if x not in exclude_canyon]

class GPSPosition_continuous_lospos_convR_onlyRNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列和R作为观测，输出三维位置校正动作和九维度的R调整
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_convR_onlyRNallcorrect, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
                                              'R_noise':spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 9+3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(3)
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(3)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = obs_Vnoise_pre
            obs_noise_R = obs_Xnoise_pre
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = obs_Vnoise_cur
            obs_noise_R = obs_Xnoise_cur

        obs_noise_R, obs_noise_Q = self._normalize_noise(obs_noise_R, obs_noise_Q)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'R_noise': obs_noise_R.reshape(1, 9, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 9 + 3])
        predict_R_noise = self.measurement_noise_scale * np.array([[action[0, 0], action[0, 1], action[0, 2]],
                                                                   [action[0, 3], action[0, 4], action[0, 5]],
                                                                   [action[0, 6], action[0, 7], action[0, 8]]])
        predict_x = action[0,9]*self.continuous_action_scale
        predict_y = action[0,10]*self.continuous_action_scale
        predict_z = action[0,11]*self.continuous_action_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        v_wls = np.array([v_x, v_y, v_z])
        action_pre = np.array([predict_x, predict_y, predict_z])
        velocity = np.sqrt(np.sum(v_wls ** 2))
        action_value = np.sqrt(np.sum(action_pre ** 2))
        if self.trajdata_range[0] == self.trajdata_range[-1]:  # set the bound of the policy for testing
            if (velocity > 40) or (PR_mean>1e6):
                predict_R_noise = predict_R_noise * 0
                x_wls = np.array([obs_x, obs_y, obs_z ])
            else:
                x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z]) # 修正了spp的结果

        rl_x, rl_y, rl_z, HPL = self.RL4KFGSDC(x_wls, v_wls, predict_R_noise, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['satnum']] = satnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['velocity']] = velocity

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = 10-np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, predict_R_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv * 10 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv *  10 ** (utc_cur - utc_pre - 1)  # process noise cov

        if self.conv_corr == 'conv_corr_1':  # 使用固定值作为初值
            R = sigma_x ** 1.250 * np.eye(3) + predict_R_noise
        elif self.conv_corr == 'conv_corr_2':
            R = self.covx + predict_R_noise

        if (utc_cur-utc_pre) < interrupt_time: # 如果时间中断，重新初始化
        ############ KF: Prediction step
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
            d = dist_err_XYZ(zs, H @ x)
        ############# KF: Update step #########
            if d < sigma_mahalanobis:
                y = zs.T - H @ x
                S = (H @ self.P) @ H.T + R
                K = (self.P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                self.P = (I - (K @ H)) @ self.P
                self.covx = R
            else:
                self.P = self.P + 10 ** 2 * Q
        else:
            x = zs
            self.P = self.P + sigma_x ** 1.250 * np.eye(3) * 100 # edict in 250623

        # predict HPL
        HPL = HPL_predict(self.P,x)

        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp']])

        self.v_pre = us
        return x[0],x[1],x[2],HPL

class GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列和R作为观测，输出三维位置校正动作和九维度的R调整，删除R输入，pos改为差分，正则化做了调整
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * (self.pos_num-1)), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 9+3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
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

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=np.diff(obs, n=1, axis=-1) # 位置差分
        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        obs_all = {'pos': obs.reshape(1, 3 * (self.pos_num-1), order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 9 + 3])
        predict_R_noise = self.measurement_noise_scale * np.array([[action[0, 0], action[0, 1], action[0, 2]],
                                                                   [action[0, 3], action[0, 4], action[0, 5]],
                                                                   [action[0, 6], action[0, 7], action[0, 8]]])
        predict_x = action[0,9]*self.continuous_action_scale
        predict_y = action[0,10]*self.continuous_action_scale
        predict_z = action[0,11]*self.continuous_action_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        if selfvel:
            v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_self']
            v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_self']
            v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_self']
        else:
            v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
            v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
            v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        v_wls = np.array([v_x, v_y, v_z])
        velocity = np.sqrt(np.sum(v_wls ** 2))
        # if self.trajdata_range[0] == self.trajdata_range[-1]:  # set the bound of the policy for testing
        #     if (velocity > 40) or (PR_mean>1e6):
        #         predict_R_noise = predict_R_noise * 0
        #         x_wls = np.array([obs_x, obs_y, obs_z ])
        #     else:
        #         x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z]) # 修正了spp的结果
        # else:
        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z])

        rl_x, rl_y, rl_z, HPL = self.RL4KFGSDC(x_wls, v_wls, predict_R_noise, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['satnum']] = satnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['velocity']] = velocity

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = 10-np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0
        elif self.reward_setting == 'RMSEratio_kf':
            reward = 10 * (np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))  -
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)))/np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, predict_R_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv # * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv # *  10 ** (utc_cur - utc_pre - 1)  # process noise cov

        if self.conv_corr == 'conv_corr_1':  # 使用固定值作为初值
            R = sigma_x ** 1.250 * np.eye(3) + predict_R_noise
        elif self.conv_corr == 'conv_corr_2':
            R = self.covx + predict_R_noise

        if (utc_cur-utc_pre) < interrupt_time: # 如果时间中断，重新初始化
        ############ KF: Prediction step
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
            d = dist_err_XYZ(zs, H @ x)
        ############# KF: Update step #########
            if d < sigma_mahalanobis:
                y = zs.T - H @ x
                S = (H @ self.P) @ H.T + R
                K = (self.P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                self.P = (I - (K @ H)) @ self.P
                self.covx = R
            else:
                self.P = self.P + 10 ** 2 * Q
        else:
            x = zs
            self.P = self.P + sigma_x ** 1.250 * np.eye(3) * 100 # edict in 250623

        # predict HPL
        HPL = HPL_predict(self.P,x)
        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp']])

        self.v_pre = us
        return x[0],x[1],x[2],HPL

class GPSPosition_continuous_lospos_PVcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列和R作为观测，输出六维位置校正动作修正位置和速度
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_Paction_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 continuous_Vaction_scale, conv_corr):
        super(GPSPosition_continuous_lospos_PVcorrect, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.conv_corr = conv_corr # 动作类型
        self.continuous_Paction_scale = continuous_Paction_scale
        self.continuous_Vaction_scale = continuous_Vaction_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 6), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 6])

        predict_x = action[0,0]*self.continuous_Paction_scale
        predict_y = action[0,1]*self.continuous_Paction_scale
        predict_z = action[0,2]*self.continuous_Paction_scale
        predict_vx = action[0,3]*self.continuous_Vaction_scale
        predict_vy = action[0,4]*self.continuous_Vaction_scale
        predict_vz = action[0,5]*self.continuous_Vaction_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        v_wls = np.array([v_x+predict_vx, v_y+predict_vy, v_z++predict_vz])
        velocity = np.sqrt(np.sum(v_wls ** 2))
        # if self.trajdata_range[0] == self.trajdata_range[-1]:  # set the bound of the policy for testing
        #     if (velocity > 40) or (PR_mean>1e6):
        #         x_wls = np.array([obs_x, obs_y, obs_z ])
        #     else:

        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z]) # 修正了spp的结果
        rl_x, rl_y, rl_z, HPL = self.RL4KFGSDC(x_wls, v_wls, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['satnum']] = satnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['velocity']] = velocity

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = 10-np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv * 10 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv * 10 ** (utc_cur - utc_pre - 1)  # process noise cov

        R = self.covx

        if (utc_cur-utc_pre) < interrupt_time: # 如果时间中断，重新初始化
        ############ KF: Prediction step
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
            d = dist_err_XYZ(zs, H @ x)
        ############# KF: Update step #########
            if d < sigma_mahalanobis:
                y = zs.T - H @ x
                S = (H @ self.P) @ H.T + R
                K = (self.P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                self.P = (I - (K @ H)) @ self.P
                self.covx = R
            else:
                self.P = self.P + 10 ** 2 * Q
        else:
            x = zs
            self.P = self.P + sigma_x ** 1.250 * np.eye(3) * 100 # edict in 250623

        # predict HPL
        HPL = HPL_predict(self.P,x)

        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp']])

        self.v_pre = us
        return x[0],x[1],x[2],HPL

class GPSPosition_continuous_lospos_onlyQNcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用RL代替KF的预测阶段，并输出Q
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_onlyQNcorrect, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})
        #                      'Q_noise':spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float)

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 9+3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        # noise cov feature process
        obs_noise_Q = self.covv
        obs_noise_R = self.covx
        obs_noise_R, obs_noise_Q = self._normalize_noise(obs_noise_R, obs_noise_Q)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C')}
        #   'Q_noise': obs_noise_Q.reshape(1, 9, order='C')
        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 9 + 3])
        predict_Q_noise = self.process_noise_scale * np.array([[action[0, 0], action[0, 1], action[0, 2]],
                                                                   [action[0, 3], action[0, 4], action[0, 5]],
                                                                   [action[0, 6], action[0, 7], action[0, 8]]])
        predict_x = action[0,9]*self.continuous_action_scale
        predict_y = action[0,10]*self.continuous_action_scale
        predict_z = action[0,11]*self.continuous_action_scale

        if self.trajdata_range[0] == self.trajdata_range[-1]:  # set the bound of the policy for testing
            if PR_mean > 1e6:
                policy = False

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        x_wls = np.array([obs_x, obs_y, obs_z]) # 修正了spp的结果
        v_wls = np.array([v_x + predict_x, v_y + predict_y, v_z + predict_z])
        rl_x, rl_y, rl_z, HPL = self.RL4KFGSDC(x_wls, v_wls, predict_Q_noise, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['satnum']] = satnum

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = 10-np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, predict_Q_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv * 100 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv * 10 ** (utc_cur - utc_pre - 1)  # process noise cov

        if self.conv_corr == 'conv_corr_1':  # 使用固定值作为初值
            Q = Q + predict_Q_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = Q + predict_Q_noise
            self.covv = Q

        R = sigma_x ** 1.250 * np.eye(3)

        if (utc_cur-utc_pre) < interrupt_time: # 如果时间中断，重新初始化
        ############ KF: Prediction step
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
            d = dist_err_XYZ(zs, H @ x)
        ############# KF: Update step #########
            if d < sigma_mahalanobis:
                y = zs.T - H @ x
                S = (H @ self.P) @ H.T + R
                K = (self.P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                self.P = (I - (K @ H)) @ self.P
                self.covx = R
            else:
                self.P = self.P + 10 ** 2 * Q
        else:
            x = zs
            self.P = self.P + R * 100 # edict in 250623

        # predict HPL
        HPL = HPL_predict(self.P,x)

        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp']])

        self.v_pre = us
        return x[0],x[1],x[2],HPL

class GPSPosition_continuous_lospos_convR_onlyRandKFcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列和R作为观测，输出九维度的R调整，最后调整KF三维位置
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_convR_onlyRandKFcorrect, self).__init__()
        self.max_visible_sat=13
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
                                              'R_noise':spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 9+3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(3)
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(3)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = obs_Vnoise_pre
            obs_noise_R = obs_Xnoise_pre
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = obs_Vnoise_cur
            obs_noise_R = obs_Xnoise_cur

        obs_noise_R, obs_noise_Q = self._normalize_noise(obs_noise_R, obs_noise_Q)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'R_noise': obs_noise_R.reshape(1, 9, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 9 + 3])
        predict_R_noise = self.measurement_noise_scale * np.array([[action[0, 0], action[0, 1], action[0, 2]],
                                                                   [action[0, 3], action[0, 4], action[0, 5]],
                                                                   [action[0, 6], action[0, 7], action[0, 8]]])
        predict_x = action[0,9]*self.continuous_action_scale
        predict_y = action[0,10]*self.continuous_action_scale
        predict_z = action[0,11]*self.continuous_action_scale

        # if self.trajdata_range[0] == self.trajdata_range[-1]:  # set the bound of the policy for testing
        #     if PR_mean > 1e6:
        #         policy = False

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        x_wls = np.array([obs_x, obs_y, obs_z]) # 修正了spp的结果
        v_wls = np.array([v_x, v_y, v_z])
        rl_x, rl_y, rl_z, HPL = self.RL4KFGSDC(x_wls, v_wls, predict_R_noise, policy) # 自适应KF
        rl_x = rl_x + predict_x
        rl_y = rl_y + predict_y
        rl_z = rl_z + predict_z

        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['satnum']] = satnum

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, predict_R_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv * 1000 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv * 10 ** (utc_cur - utc_pre - 1)  # process noise cov

        if self.conv_corr == 'conv_corr_1':  # 使用固定值作为初值
            R = sigma_x ** 1.250 * np.eye(3) + predict_R_noise
        elif self.conv_corr == 'conv_corr_2':
            R = self.covx + predict_R_noise

        if (utc_cur-utc_pre) < interrupt_time: # 如果时间中断，重新初始化
        ############ KF: Prediction step
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
            d = dist_err_XYZ(zs, H @ x)
        ############# KF: Update step #########
            if d < sigma_mahalanobis:
                y = zs.T - H @ x
                S = (H @ self.P) @ H.T + R
                K = (self.P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                self.P = (I - (K @ H)) @ self.P
                self.covx = R
            else:
                self.P = self.P + 10 ** 2 * Q
        else:
            x = zs
            self.P = self.P + sigma_x ** 1.250 * np.eye(3) * 100 # edict in 250623

        # predict HPL
        HPL = HPL_predict(self.P,x)

        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp'],
                 self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp']])

        self.v_pre = us
        return x[0],x[1],x[2],HPL

class GPSPosition_continuous_lospos_correction(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        只对spp结果修正，不进行kf
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_correction, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * self.feature_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'full_nocanyon':
            self.tripIDlist = traj_full_nocanyon
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        try:
            feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        except:
            pass
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]
        if len(sorted_feature_tmp) < self.max_visible_sat:
            obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp
        else:
            sorted_feature_tmp = sorted_feature_tmp[sorted_feature_tmp[:, 4].argsort()]
            sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat,:]
            obs_feature[0:self.max_visible_sat, :] = sorted_feature_tmp

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        EA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 3])
        predict_x = action[0,0]*self.continuous_action_scale
        predict_y = action[0,1]*self.continuous_action_scale
        predict_z = action[0,2]*self.continuous_action_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z]) # 修正了spp的结果
        rl_x, rl_y, rl_z = x_wls[0],x_wls[1],x_wls[2]

        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        error_bl = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))
        error_kf = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2))
        tripID = self.tripIDlist[self.tripIDnum]

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = EA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['satnum']] = satnum

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error, 'error_bl':error_bl,
                                   'error_kf':error_kf,'tripid': self.tripIDnum,'tripID':tripID} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')


def HPL_predict(P_ecef,ecef_position):
    """
    估计水平定位标准差
    Args:
        P_ecef: KF估计协方差
        ecef_position: ecef下定位坐标
    Returns: 水平保护限HPL
    """
    x, y, z = ecef_position

    # 计算ENU轴的单位向量
    # east = np.array([-y, x, 0]) / np.sqrt(x ** 2 + y ** 2)
    # north = np.array([-x * z, -y * z, x ** 2 + y ** 2]) / np.sqrt((x ** 2 + y ** 2) * (x ** 2 + y ** 2 + z ** 2))
    # up = np.array([x, y, z]) / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # # 构建旋转矩阵
    # R = np.vstack([east, north, up])
    # # 转换协方差矩阵
    # P_enu = R @ P_ecef @ R.T
    # # 提取水平分量
    # HPL_enu = 4.29 * sqrt(P_enu[0,0]+P_enu[1,1])

    # 转换至大地坐标系
    llh_pre = coord.ecef2geodetic(ecef_position)
    H = np.array([[-math.sin(llh_pre[0]),math.cos(llh_pre[0]),0],
                  [-math.sin(llh_pre[1])*math.cos(llh_pre[0]),-math.sin(llh_pre[1])*math.sin(llh_pre[0]),math.cos(llh_pre[1])],
                  [math.cos(llh_pre[1])*math.cos(llh_pre[0]),math.cos(llh_pre[1])*math.sin(llh_pre[0]),math.sin(llh_pre[1])]])
    P_llh = H @ P_ecef @ H.T
    HPL_llh = k_xy * sqrt(P_llh[0,0]+P_llh[1,1])

    return HPL_llh

class GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列和R作为观测，输出三维位置校正动作，对R的调整只有对角线
        trajdata_range: 轨迹ID范围
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                 noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0EA_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0EA_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
                                              'R_noise':spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3+3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3) # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime'] # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values)-1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp_smoothing']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp_smoothing']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp_smoothing']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs #self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        # gnss1 = gnss
        gnss[:,1]=(gnss[:,1]) / res_max
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        gnss[:,5]=(gnss[:,5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:,6]=(gnss[:,6]) / max(EA_max, np.abs(EA_min))

        # gnss[:, 1] = ((gnss[:,1]) /max(np.abs(gnss[:,1])))
        # gnss[:, 2] = ((gnss[:,2]) /max(np.abs(gnss[:,2])))
        # gnss[:, 3] = ((gnss[:,3]) /max(np.abs(gnss[:,3])))
        # gnss[:, 4] = ((gnss[:,4]) /max(np.abs(gnss[:,4])))
        # gnss[:, 5] = ((gnss[:,5]) /max(np.abs(gnss[:,5])))
        # gnss[:, 6] = ((gnss[:,6]) /max(np.abs(gnss[:,6])))
        return gnss

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp_smoothing']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp_smoothing']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp_smoothing']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        try:
            feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
        except:
            pass
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5, 6] # 伪距 los CN0 高度角
        BDSI1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]  # only GPSL1
        sorted_feature_tmp = sorted_feature_tmp[BDSI1_index, :]  # only GPSL1
        obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(3)
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(3)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = obs_Vnoise_pre
            obs_noise_R = obs_Xnoise_pre
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = obs_Vnoise_cur
            obs_noise_R = obs_Xnoise_cur

        obs_noise_R, obs_noise_Q = self._normalize_noise(obs_noise_R, obs_noise_Q)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'R_noise': np.diag(obs_noise_R).reshape(1, 3, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        HA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction
        action = np.reshape(action, [1, 3 + 3])
        predict_R_noise = self.measurement_noise_scale * np.array([[action[0, 0], 0, 0],
                                                                   [0, action[0, 1], 0],
                                                                   [0, 0, action[0, 2]]])
        predict_x = action[0,3]*self.continuous_action_scale
        predict_y = action[0,4]*self.continuous_action_scale
        predict_z = action[0,5]*self.continuous_action_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp_smoothing']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp_smoothing']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp_smoothing']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z]) # 修正了spp的结果
        v_wls = np.array([v_x, v_y, v_z])
        rl_x, rl_y, rl_z = self.RL4KFGSDC(x_wls, v_wls, predict_R_noise, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['HA_mean']] = HA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KFGSDC(self,zs, us, predict_R_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        ############ KF: Prediction step
        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = self.covv * 100 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = self.covv * 10 ** (utc_cur - utc_pre - 1)  # process noise cov
        if (utc_cur-utc_pre) < interrupt_time:
            x = F @ x + us.T * (utc_cur-utc_pre)
            self.P = (F @ self.P) @ F.T + Q
        ############# KF: Update step
            if self.conv_corr == 'conv_corr_1': # 使用固定值作为初值
                R = sigma_x ** 1.250 * np.eye(3) + predict_R_noise
            elif self.conv_corr == 'conv_corr_2':
                R = self.covx + predict_R_noise
            y = zs.T - H @ x
            S = (H @ self.P) @ H.T + R
            K = (self.P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            self.P = (I - (K @ H)) @ self.P
            self.covx = R
        else:
            x = zs

        # if self.current_step % 30 != 0: # 每20步用初始值初始化一次矩阵
        if policy == False: # if bound, initial the state of KF
            x = np.array(
                [self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf'],
                 self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf'],
                 self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']])

        self.v_pre = us
        return x[0],x[1],x[2]

class GPSPosition_continuous_lospos_convQR_QRNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        使用GNSS测量数据以及pos序列作为观测，输出三维位置校正动作
        trajdata_range: 轨迹ID范围，这里只有一条
        traj_type：每条轨迹数据在这条轨迹的使用范围，例[0,0.7]即使用该轨迹定位点0到0.7范围的数据点构建环境
        triptype：数据类型，这里只有一组数据20241112Haige
        continuous_action_scale：动作尺度限制
        continuous_actionspace：动作范围
        reward_setting：奖励设置
        baseline_mod：基线方法 spp rtk
        traj_len：pos长度
        noise_scale_dic：噪声协方差矩阵修正动作尺度
        conv_corr：修正策略
    """
    # def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    def __init__(self,trajdata_range, traj_type, triptype, continuous_Xaction_scale, continuous_Vaction_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic, conv_corr):
        super(GPSPosition_continuous_lospos_convQR_QRNallcorrect, self).__init__()
        self.max_visible_sat=24
        self.feature_num = CN0EA_num
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0EA_num)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
                                              'Q_noise':spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float),
                                              'R_noise':spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float)})

        if triptype == 'full':
            self.tripIDlist = traj_full
        if triptype == 'openroad':
            self.tripIDlist = traj_openroad
        if triptype == 'canyon':
            self.tripIDlist = traj_canyon
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        if triptype == 'forest':
            self.tripIDlist = traj_forest
        if triptype == 'overpass':
            self.tripIDlist = traj_overpass
        if triptype == 'no_canyon':
            self.tripIDlist = traj_openroad+traj_highway+traj_forest+traj_overpass

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process'] # 过程噪声协方差尺度
        self.measurement_noise_scale = noise_scale_dic['measurement'] # 测量噪声协方差
        self.conv_corr = conv_corr # 动作类型
        self.continuous_action_scale = continuous_Xaction_scale
        self.continuous_Vaction_scale = continuous_Vaction_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 24), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        data_truth_dic[self.tripIDlist[self.tripIDnum]] = data_truth_dic[
            self.tripIDlist[self.tripIDnum]].copy().reset_index(drop=True)
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature[self.tripIDlist[self.tripIDnum]].copy()
        # set parameter for KF
        self.P = initial_P ** 2 * np.eye(3)  # 协方差矩阵
        self.covv = sigma_v ** 1.250 * np.eye(3)
        self.covx = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
        self.datatime = self.baseline['UTCtime']  # UnixTimeMillis
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UTCtime'].values) - 1, 'UTCtime']

        # gen pred
        if 'spp' in self.baseline_mod:
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_spp_smoothing']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_spp_smoothing']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_spp_smoothing']
        elif self.baseline_mod == 'rtk':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_rtk']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_rtk']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_rtk']

        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs = self._next_observation()
        # reset the previous velocity
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 2), 'VZEcefMeters_wls']
        self.v_pre = np.array([v_x, v_y, v_z])

        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / xecef_normal
        state[1] = state[1] / yecef_normal
        state[2] = state[2] / zecef_normal
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

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / (sigma_x ** 1.250)
        obs_noise_Q = obs_noise_Q / (sigma_v ** 1.250)
        # zero-score normalize
        return obs_noise_R, obs_noise_Q

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if 'spp' in self.baseline_mod:
            obs = np.append(obs,
                            [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_spp_smoothing']],
                             [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_spp_smoothing']],
                             [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_spp_smoothing']]],axis=1)
        elif self.baseline_mod == 'rtk':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_rtk']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_rtk']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_rtk']]], axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp= self._normalize_los(feature_tmp)
        sorted_indices = np.argsort(-feature_tmp[:, 5])  # 注意这里的负号，用于从大到小排序
        sorted_feature_tmp = feature_tmp[sorted_indices]
        sorted_feature_tmp = sorted_feature_tmp[:self.max_visible_sat]
        obs_feature = np.zeros([(self.max_visible_sat), self.feature_num])
        # for i in range(len(self.visible_sat)):
        #     # if self.visible_sat[i] in feature_tmp[:,0]:
        #     if self.visible_sat[i] in feature_tmp[:, 0]:
        #         obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        feature_index_list = [1, 2, 3, 4, 5, 6] # 伪距 los CN0 高度角
        GPSL1_index = sorted_feature_tmp[:, 0] < 100
        sorted_feature_tmp = sorted_feature_tmp[:, feature_index_list]  # only GPSL1
        sorted_feature_tmp = sorted_feature_tmp[GPSL1_index, :]  # only GPSL1
        obs_feature[0:len(sorted_feature_tmp), :] = sorted_feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(3)
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(3)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = obs_Vnoise_pre
            obs_noise_R = obs_Xnoise_pre
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = obs_Vnoise_cur
            obs_noise_R = obs_Xnoise_cur

        obs_noise_R, obs_noise_Q = self._normalize_noise(obs_noise_R, obs_noise_Q)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),'gnss': obs_feature.reshape(1, self.feature_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, 9, order='C'), 'R_noise': obs_noise_R.reshape(1, 9, order='C')}

        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UTCtime'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UTCtime']  # UnixTimeMillis

        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'].copy()
        satnum = len(feature_tmp)
        CN0_mean = np.nanmean(feature_tmp[:, 5])
        HA_mean = np.nanmean(feature_tmp[:, 6])
        PR_mean = np.nanmean(np.abs(feature_tmp[:, 1]))
        policy = True
        # action for new prediction

        action = np.reshape(action, [1, 24])
        predict_Q_noise = self.process_noise_scale * np.array([[action[0,0], action[0,1], action[0,2]],
                                   [action[0,3], action[0,4], action[0,5]],
                                    [action[0,6], action[0,7], action[0,8]]])
        predict_Vx = action[0,9]*self.continuous_Vaction_scale
        predict_Vy = action[0,10]*self.continuous_Vaction_scale
        predict_Vz = action[0,11]*self.continuous_Vaction_scale

        predict_R_noise = self.measurement_noise_scale * np.array([[action[0,12], action[0,13], action[0,14]],
                                   [action[0,15], action[0,16], action[0,17]],
                                    [action[0,18], action[0,19], action[0,20]]])
        predict_x = action[0,21]*self.continuous_action_scale
        predict_y = action[0,22]*self.continuous_action_scale
        predict_z = action[0,23]*self.continuous_action_scale

        if  'spp' in self.baseline_mod:
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_spp_smoothing']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_spp_smoothing']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_spp_smoothing']
        elif self.baseline_mod == 'rtk':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_rtk']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_rtk']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_rtk']

        kf_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']  # 实时KF作为baselime
        kf_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
        kf_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        v_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls']
        v_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls']
        v_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls']

        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX_gt']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY_gt']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ_gt']

        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z])
        v_wls = np.array([v_x + predict_Vx, v_y + predict_Vy, v_z + predict_Vz])
        rl_x, rl_y, rl_z = self.RL4KF_Haige(x_wls, v_wls, predict_R_noise, predict_Q_noise, policy) # 自适应KF
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['CN0_mean']] = CN0_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['EA_mean']] = HA_mean
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['PR_mean']] = PR_mean

        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline, 'error':error} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

    def RL4KF_Haige(self,zs, us, predict_R_noise, predict_Q_noise, policy=True):
        # Parameters
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num-2), 'X_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Y_RLpredict'],
             self.baseline.loc[self.current_step + (self.pos_num-2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)
        utc_cur = self.datatime[self.current_step + (self.pos_num - 1)]
        utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]

        # KF: Prediction step
        ## Estimated WLS velocity covariance
        if self.conv_corr == 'conv_corr_1':
            Q = sigma_v ** 1.250 * np.eye(3) + predict_Q_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = self.covv + predict_Q_noise

        if np.sqrt(np.sum(us ** 2)) > outlier_velocity:  # check outlier of velocity
            us = self.v_pre
            Q = Q * 100 * 10 ** (utc_cur - utc_pre - 1)
        else:
            Q = Q * 10 ** (utc_cur - utc_pre - 1)  # process noise cov

        x = F @ x + us.T
        self.P = (F @ self.P) @ F.T + Q
        d = distance.mahalanobis(zs, H @ x, np.linalg.inv(self.P))
        # KF: Update step
        if self.conv_corr == 'conv_corr_1': # 使用固定值作为初值
            R = sigma_x ** 1.250 * np.eye(3) + predict_R_noise
        elif self.conv_corr == 'conv_corr_2':
            R = self.covx + predict_R_noise
        if (utc_cur - utc_pre) < interrupt_time:
            y = zs.T - H @ x
            S = (H @ self.P) @ H.T + R
            K = (self.P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            self.P = (I - (K @ H)) @ self.P
            self.covx = R
        else:
            x = zs

        if policy == False:  # if bound, initial the state of KF
            x = np.array(
                [self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf'],
                 self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf'],
                 self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']])
        return x[0],x[1],x[2]