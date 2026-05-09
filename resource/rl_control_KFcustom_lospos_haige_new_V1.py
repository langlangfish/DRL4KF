# coding=utf-8
"""
@Project:   MAML2
@File:      rl_control_custom_lospos_haige_new_V1.py
@Author:    Tang
@Date:      2025/07/15 17:23
@IDE:       PyCharm 
FUNCTION:
    用于海格最新模组数据训练
"""
import numpy as np
import pandas as pd
# from env.GSDC_2022_LOSPOS import *
from env.HaigeRLAKF_lospos_new_env_V1 import *  # RL环境
# from env.GSDC_2022_LOS import *
# from env.GSDC_2022_ECEF import *
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from model.a2c import A2C
from model.ppo import PPO
# from model.ppo_recurrent import RecurrentPPO
from model.ppo_recurrent_ATF1_AKF import RecurrentPPO
from env.env_param import *
from funcs.haige_utilis import *
from model.model_ATF_KF import *
#from funcs.PPO_SR import *
from collections import deque
import time
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 显卡
running_date = 'RLAKF_20250903_alltraining_selfvel' # RLAKF_20250715_after0620training RLAKF_20250802_alltraining
moretests = True  # True False
# 参数调整
haige_train_step_num = 20000 # default=25000
haige_learning_rate_list = [1e-4]
triptype = 'full_nocanyon' # full full_nocanyon
networkmod = 'continuous_lstm' # 该版本只选择continuous_lstm
envmod = 'losposcovR_onlyRNallcorrect_v2'
"""
参数说明
# running_date：文件夹保存日期
# haige_train_step_num（次要参数）：训练迭代次数
# haige_learning_rate_list（次要参数）：学习率
# envmode
    1. losposcovR_onlyRNallcorrect: 使用LOS POS R作为输入来学R和修正SPP再做KF
    2. losposcovRDIA_onlyRNallcorrect: R的调整动作使用对角矩阵
    3. lospos_onlycorrect： 使用LOS和POS修正SPP
    4. lospos_convR_onlyRandKFcorrect: 学R做KF最后修正
    5. lospos_onlyQNcorrect: 学Q和速度修正再KF
    6. lospos_PVcorrect: 6维度动作只修正位置速度
    7. losposcovR_onlyRNallcorrect_v2: 删除R输入，pos改为差分输入，正则化做了调整
# conv_corr: 
    1. conv_corr_1 为在固定的数值上调整（一般选这个），2. conv_corr_2 为在前一次数值上调整
# domain_select：目前的数据划分默认使用域内测试，即每个traj的0.9作为训练，余下0.1作为测试
# noise_scale_dic（主要参数）：'process'：过程噪声协方差Q的尺度（一般不用），'measurement'：测量噪声协方差R尺度
# continuous_action_scale（主要参数）：SPP位置修正动作尺度调整
# reward_setting（一般不用动）：奖励函数设置
# need_RBO：是否使用RBO算子增加泛化，但会增加训练时间，而且好像没什么提升
# ent_coef（次要参数）：熵正则项的系数，可以增加策略探索
# moreteststypelist（一般不动）：测试场景
"""
noise_scale_dic = {'process':1e-4,'measurement':1e-3} # 调整协方差的动作尺度，该版本只需要调整'measurement'
conv_corr = 'conv_corr_1' # 调整协方差方式： conv_corr_1 conv_corr_2，该版本选择conv_corr_1会好
continuous_action_scale = 20e-1  # 修正spp位置动作的尺度
continuous_Vaction_scale = 8e-1  # 修正spp速度动作的尺度
ent_coef = 0.01 # 熵正则参数

network_unit = 128 # MLP网络隐藏层个数（policy,value）
laynum = 2 # 层数
trajdata_sort = 'sorted'  # 'randint' 'sorted'
baseline_mod = 'spp'  # baseline方法  spp kf
traj_len = 5 # pos长度设置
reward_setting = 'RMSEadv_kf'  # select reward: 'RMSE' ‘RMSEadv' 'RMSEadv_kf',RMSEratio_kf
domain_select = 'indomain'   # indomain:域内测试 outdomain：域外测试
# use RBO for improve generalization, but it seems not effective
need_RBO = False
noisescale = 0.01 # 0.4,0.1,0.05,0.01,0.005
beta_robust = 0.01

detail = f'ent={ent_coef}_sv={sigma_v}' # 文件夹命名补充细节

if domain_select == 'indomain': # 所有轨迹训练，取前80%的点，后20%测试
    traj_type_train = [0, 0.8] #
    traj_type_test = [0.8, 1]
    ratio = 1
elif domain_select == 'outdomain': # 轨迹所有点训练，取前ratio条轨迹作为训练
    traj_type_train = [0, 1]
    traj_type_test = [0, 1]
    ratio = 0.9

# select network and environment    选择网络和环境
discrete_lists = ['discrete', 'discrete_A2C', 'discrete_lstm', 'ppo_discrete']
continuous_lists = ['continuous', 'continuous_lstm', 'continuous_lstm_custom', 'ppo', 'continuous_custom']
# continuous action settings 连续动作设置
max_action = 100  # 最大动作的范围
continuous_actionspace = [-max_action, max_action]

# Setting for training and testing data
if triptype == 'full':
    tripIDlist = traj_full
elif triptype == 'full_nocanyon':
    tripIDlist = traj_full_nocanyon

moreteststypelist = ['openroad','canyon','overpass','highway','forest']
# 测试环境：'openroad','canyon','overpass','highway','forest'
trajdata_range= [0,int(np.ceil((len(tripIDlist)-1)*ratio))] # 测试范围

############ 设置测试参数
"""
参数说明
# onlytesting：设置为False的时候模型才训练，True的话会调用已训练的模型参数进行测试
# testdate：指定测试的文件夹
"""
onlytesting = False # only test no learn.yaml，设置为false才会训练，注意
posnum_test = 5 # 测试的pos序列
testdate = 'RLAKF_20250830_alltraining_selfvel/full_nocanyon_spp_losposcovR_onlyRNallcorrect_v2_indomain_RMSEadv_kf'
if onlytesting:
    model_summary = True # 打印模型结构
    model_basefolder=f'{dir_path}/records_values/{testdate}'
    model_folderlist=os.listdir(model_basefolder)
    model_folderlist.sort()
    # model_folderlist = [ 'lr=0.0001_20000_size_128x2_XAS=2.0_RS=0.1_conv_corr_1_ent=0.01'] # only for testing

# trajdata_range = [5,5]
if onlytesting == False:
    for haige_learning_rate in haige_learning_rate_list:
        # calculate the range for target and source
        if networkmod in discrete_lists:
            print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
        elif networkmod in continuous_lists:  # 打印模式信息
            print(
                f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')

        QS, RS = noise_scale_dic['process'], noise_scale_dic['measurement']
        if need_RBO:
            tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{baseline_mod}_{envmod}_{domain_select}/' \
                          f'lr={haige_learning_rate}_{haige_train_step_num}_size_{network_unit}x{laynum}_XAS={continuous_action_scale}_RS={RS}_{conv_corr}_NS={noisescale}'
        else:
            tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{baseline_mod}_{envmod}_{domain_select}_{reward_setting}/' \
                        f'lr={haige_learning_rate}_{haige_train_step_num}_size_{network_unit}x{laynum}_XAS={continuous_action_scale}_RS={RS}_{conv_corr}_{detail}'

        if envmod == 'losposcovR_onlyRNallcorrect':
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'losposcovR_onlyRNallcorrect_v2':
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'losposcovRDIA_onlyRNallcorrect':
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'lospos_onlycorrect':
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_correction(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace, reward_setting,trajdata_sort, baseline_mod, traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'lospos_convR_onlyRandKFcorrect':
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_convR_onlyRandKFcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace, reward_setting,trajdata_sort, baseline_mod, traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'lospos_onlyQNcorrect':
            tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{baseline_mod}_{envmod}_{domain_select}_{reward_setting}/' \
                        f'lr={haige_learning_rate}_{haige_train_step_num}_size_{network_unit}x{laynum}_VAS={continuous_action_scale}_QS={QS}_{conv_corr}_{detail}'
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_onlyQNcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace, reward_setting,trajdata_sort, baseline_mod, traj_len, noise_scale_dic, conv_corr)])
        elif envmod == 'lospos_PVcorrect':
            tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{baseline_mod}_{envmod}_{domain_select}/' \
                          f'lr={haige_learning_rate}_{haige_train_step_num}_size_{network_unit}x{laynum}_XAS={continuous_action_scale}_VAS={continuous_Vaction_scale}_{conv_corr}'
            env = DummyVecEnv(
                [lambda: GPSPosition_continuous_lospos_PVcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale,
                                                       continuous_actionspace, reward_setting,trajdata_sort, baseline_mod, traj_len, continuous_Vaction_scale, conv_corr)])

        if networkmod == 'continuous_lstm':
            obs = env.reset()  # 模型是ppo算法
            net_arch = [dict(pi=[network_unit for _ in range(laynum)], vf=[network_unit for _ in range(laynum)])]
            policy_kwargs = dict(net_arch=net_arch)
            if need_RBO:
                from model.ppo_recurrent_ATF1_AKF_RBO import RecurrentPPO
                model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=2, tensorboard_log=tensorboard_log,
                                     learning_rate=haige_learning_rate, policy_kwargs=policy_kwargs,noisescale=noisescale,
                                     beta_robust=beta_robust,ent_coef=ent_coef)
            else:
                model = RecurrentPPO("MultiInputLstmPolicy", env, verbose = 2, tensorboard_log = tensorboard_log,
                                     learning_rate = haige_learning_rate,policy_kwargs=policy_kwargs,ent_coef=ent_coef)
            model.learn(total_timesteps = haige_train_step_num, eval_log_path = tensorboard_log)


        elif networkmod == 'continuous_lstmATF1':
            obs = env.reset()  # 模型是ppo算法
            features_dim_gnss = obs['gnss'].shape[-1]
            features_dim_pos = obs['pos'].shape[-1]
            features_dim_R = obs['R_noise'].shape[-1]
            R_encoder = CustomATF1_AKFRL_losposcovR
            net_arch = [network_unit for _ in range(laynum)]
            dim_R = features_dim_gnss + features_dim_pos + features_dim_R
            policy_kwargs_R = dict(features_extractor_class=R_encoder,  # CustomCNN CustomMLP
                                   features_extractor_kwargs=dict(features_dim=dim_R), ATF_trig=networkmod,net_arch=net_arch)
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, policy_kwargs=policy_kwargs_R,
                                 tensorboard_log=tensorboard_log, learning_rate=haige_learning_rate, seed=None)
            model.learn(total_timesteps=haige_train_step_num, eval_log_path=tensorboard_log)

        model.save(model.logger.dir +
                   f'/{networkmod}_{haige_learning_rate}_{haige_train_step_num}.pth')
        logdirname = model.logger.dir + f'/haige_{haige_learning_rate}_{haige_train_step_num}'
        recording_results_ecef(data_truth_dic, trajdata_range, tripIDlist, logdirname, baseline_mod,traj_record=False)
        print('Model training finished.')

        # more tests
        if moretests:
            all_type_rl_3derr = []
            all_type_or_3derr = []
            all_type_rl_llerr = []
            all_type_or_llerr = []
            # 'openroad','canyon','overpass','highway','forest'
            for testtype in moreteststypelist:
                print(f'More test for {testtype} env begin here')
                if testtype == 'openroad':
                    tripIDlist_test = traj_openroad
                elif testtype == 'canyon':
                    tripIDlist_test = traj_canyon
                elif testtype == 'overpass':
                    tripIDlist_test = traj_overpass
                elif testtype == 'highway':
                    tripIDlist_test = traj_highway
                elif testtype == 'forest':
                    tripIDlist_test = traj_forest

                if domain_select == 'indomain':
                    more_test_trajrange = [0,len(tripIDlist_test)-1] # 默认用所有轨迹ID [0, len(tripIDlist) - 1]
                elif domain_select == 'outdomain':
                    if triptype == 'full_nocanyon' and testtype == 'canyon':
                        more_test_trajrange = [0, len(tripIDlist_test) - 1]
                    elif testtype == 'highway':
                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * 0.5)) + 1, len(tripIDlist_test) - 1]
                    else:
                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1, len(tripIDlist_test) - 1]

                if (testtype == triptype) or (triptype=='full'):
                    traj_type = traj_type_test  # 独立同分布测试
                elif triptype=='full_nocanyon':
                    traj_type = traj_type_test
                else:
                    traj_type = [0, 1]  # 不同分布测试范围

                test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                # test_trajlist = [5,5]
                for test_traj in test_trajlist:
                    test_trajdata_range = [test_traj, test_traj]

                    if networkmod in discrete_lists:
                        env = DummyVecEnv(
                            [lambda: GPSPosition_discrete_ecef(test_trajdata_range, triptype, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod,
                                                               test_traj)])
                    elif networkmod in continuous_lists:
                        if envmod == 'losposcovR_onlyRNallcorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'losposcovR_onlyRNallcorrect_v2':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'losposcovRDIA_onlyRNallcorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'lospos_onlycorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_correction(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'lospos_convR_onlyRandKFcorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRandKFcorrect(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'lospos_onlyQNcorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_onlyQNcorrect(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          noise_scale_dic,conv_corr)])
                        elif envmod == 'lospos_PVcorrect':
                            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_PVcorrect(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len,
                                                          continuous_Vaction_scale,conv_corr)])

                    obs = env.reset()
                    maxiter = 100000
                    for iter in range(maxiter):
                        if iter == 0: # reset state for a perid of iterations
                            action, _states = model.predict(obs,deterministic=True)
                        else:
                            action, _states = model.predict(obs, deterministic=True, state=_states)
                        obs, rewards, done, info = env.step(action)
                        tmp = info[0]['tripIDnum']
                        if iter <= 3 or iter % np.ceil(maxiter / 10) == 0:
                            # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                            print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                        # elif rewards < -5:
                        #     rl_error = info[0]['error']
                        #     obs_error = info[0]['error_bl']
                        #     print(f'RL error: {rl_error}, Obs error: {obs_error}')
                        elif done:
                            print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                            break

                logdirname = model.logger.dir + f'/testmore_{testtype}_'
                avg_xyz_err,avg_xyz_or_err,avg_rl_llerr,avg_or_llerr = recording_results_ecef(data_truth_dic, [test_trajlist[0], test_trajlist[-1]],
                                    tripIDlist_test,logdirname, baseline_mod, traj_record = False)
                all_type_rl_3derr.append(avg_xyz_err)
                all_type_or_3derr.append(avg_xyz_or_err)
                all_type_rl_llerr.append(avg_rl_llerr)
                all_type_or_llerr.append(avg_or_llerr)
                print(f'More Test {testtype} finished.')

        record_difftype = pd.DataFrame({'Type': moreteststypelist, '3Derr': all_type_rl_3derr, 'or_3Derr': all_type_or_3derr,
             'llerr': all_type_rl_llerr, 'or_llerr': all_type_or_llerr})
        record_difftype.loc[len(record_difftype)] = ['average', np.mean(all_type_rl_3derr),np.mean(all_type_or_3derr),np.mean(all_type_rl_llerr),
                                                     np.mean(all_type_or_llerr)]
        record_difftype.to_csv(model.logger.dir + f'/record_diserr={np.mean(all_type_rl_3derr):.4}.csv', index=False)
        # with open( model.logger.dir + f'/all_llerr={np.mean(all_type_err):.2}',"w") as file:
        #     for i,testtype in enumerate(moreteststypelist):
        #         file.write(f'{testtype}:{all_type_err[i]:.2}\n')
        #     file.write(f'Average: {np.mean(all_type_err):.2}')

elif onlytesting:
    for model_folder in model_folderlist:
        # if 'lr=8e-05' in model_folder:
        #     continue
        #record model
        # if networkmod in model_folder:
        model_sepfolderlist=os.listdir(f'{model_basefolder}/{model_folder}') # PPO_1
        model_sepfolderlist.sort()
        # model_sepfolderlist=['RecurrentPPO_1','RecurrentPPO_2']
        """
        从文件夹名提取相应参数，注意测试环境的参数设置要和训练的一致
        """
        if f'lr=' in model_folder and f'{haige_train_step_num}' in model_folder:
            network_unit_test = int(model_folder.split(f'size_')[1].split('x')[0])
            laynum_test = int(model_folder.split('x')[1].split('_XAS')[0])
            continuous_action_scale_test = float(model_folder.split('XAS=')[1].split('_RS')[0])
            noise_scale_dic_test = {'process':1e-9,'measurement':float(model_folder.split('_RS=')[1].split('_conv')[0])}
        else:
            continue

        for model_sepfolder in model_sepfolderlist:
            process_trig = False
            if ('csv' not in model_sepfolder) and ('txt' not in model_sepfolder):
                model_filelist=os.listdir(f'{model_basefolder}/{model_folder}/{model_sepfolder}')
                model_filelist.sort()
                for model_file in model_filelist:
                    if '.pth' in model_file:
                        model_filename=model_file
                        process_trig = True
                        break
                    else:
                        process_trig = False

            if process_trig:
                model_loggerdir=f'{model_basefolder}/{model_folder}/{model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{model_loggerdir}, {t}')
                model_filenamepath=f'{model_loggerdir}/{model_filename}'
                if envmod == 'losposcovR_onlyRNallcorrect':
                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale_test,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
                elif envmod == 'losposcovR_onlyRNallcorrect_v2':
                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2(trajdata_range, traj_type_train, triptype, continuous_action_scale_test,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
                elif envmod == 'losposcovRDIA_onlyRNallcorrect':
                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect(trajdata_range, traj_type_train, triptype, continuous_action_scale_test,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])
                elif envmod == 'lospos_onlycorrect':
                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_correction(trajdata_range, traj_type_train, triptype, continuous_action_scale_test,
                                                       continuous_actionspace,reward_setting, trajdata_sort, baseline_mod,traj_len, noise_scale_dic, conv_corr)])

                net_arch = [dict(pi=[network_unit_test for _ in range(laynum_test)], vf=[network_unit_test for _ in range(laynum_test)])]
                policy_kwargs = dict(net_arch=net_arch)
                model = RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs)
                model.load(model_filenamepath, env=env)
                #model = RecurrentPPO.load(model_filenamepath, env=env)

                if model_summary:
                    print(model.policy)
                    # 获取模型的参数
                    params = list(model.policy.parameters())
                    # 计算参数的总大小
                    total_params = sum(p.numel() for p in params)
                    print(f"模型的总参数数量为：{total_params}")

                if moretests:
                    all_type_rl_3derr = []
                    all_type_rl_llerr = []
                    all_type_or_3derr = []
                    all_type_or_llerr = []
                    # 'openroad','canyon','overpass','highway','forest'
                    for testtype in moreteststypelist:
                        print(f'More test for {testtype} env begin here')
                        if testtype == 'openroad':
                            tripIDlist_test = traj_openroad
                        elif testtype == 'canyon':
                            tripIDlist_test = traj_canyon
                        elif testtype == 'overpass':
                            tripIDlist_test = traj_overpass
                        elif testtype == 'highway':
                            tripIDlist_test = traj_highway
                        elif testtype == 'forest':
                            tripIDlist_test = traj_forest

                        if domain_select == 'indomain':
                            more_test_trajrange = [0, len(tripIDlist_test)-1]  # 默认用所有轨迹ID [0, len(tripIDlist) - 1]
                        elif domain_select == 'outdomain':
                            if triptype == 'full_nocanyon' and testtype == 'canyon':
                                more_test_trajrange = [0, len(tripIDlist_test) - 1]
                            elif testtype == 'highway':
                                more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * 0.5)) + 1,len(tripIDlist_test) - 1]
                            else:
                                more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,len(tripIDlist_test) - 1]

                        if (testtype == triptype) or (triptype=='full'):
                            traj_type = traj_type_test  # 独立同分布测试
                        elif triptype == 'full_nocanyon':
                            traj_type = traj_type_test
                        else:
                            traj_type = [0, 1]  # 不同分布测试范围

                        test_trajlist =  range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]

                            if networkmod in discrete_lists:
                                env = DummyVecEnv(
                                    [lambda: GPSPosition_discrete_ecef(test_trajdata_range, triptype, action_scale,
                                                                       discrete_actionspace, reward_setting,
                                                                       trajdata_sort, baseline_mod,
                                                                       test_traj)])
                            elif networkmod in continuous_lists:
                                if envmod == 'losposcovR_onlyRNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect(
                                        test_trajdata_range, traj_type, testtype,
                                        continuous_action_scale_test, continuous_actionspace, reward_setting, trajdata_sort,
                                        baseline_mod, traj_len,
                                        noise_scale_dic_test, conv_corr)])
                                elif envmod == 'losposcovR_onlyRNallcorrect_v2':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convR_onlyRNallcorrect_V2(
                                        test_trajdata_range, traj_type, testtype,
                                        continuous_action_scale_test, continuous_actionspace, reward_setting, trajdata_sort,
                                        baseline_mod, traj_len,
                                        noise_scale_dic_test, conv_corr)])
                                elif envmod == 'losposcovRDIA_onlyRNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convRDIA_onlyRNallcorrect(
                                        test_trajdata_range, traj_type, testtype,
                                        continuous_action_scale_test, continuous_actionspace, reward_setting, trajdata_sort,
                                        baseline_mod, traj_len,
                                        noise_scale_dic_test, conv_corr)])
                                elif envmod == 'lospos_onlycorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_correction(
                                        test_trajdata_range, traj_type, testtype,
                                        continuous_action_scale_test, continuous_actionspace, reward_setting,
                                        trajdata_sort,baseline_mod, traj_len,
                                        noise_scale_dic_test, conv_corr)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter == 0: # reset state for a perid of iterations
                                    action, _states = model.predict(obs, deterministic=True)
                                else:
                                    action, _states = model.predict(obs, deterministic=True, state=_states)
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 3 or iter % np.ceil(maxiter / 10) == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                # elif rewards < -5:
                                #     rl_error = info[0]['error']
                                #     obs_error = info[0]['error_bl']
                                #     kf_error = info[0]['error_kf']
                                #     tripID = info[0]['tripID']
                                #     print(f'Iter {iter}, traj {tmp} reward is {rewards}, {tripID}')
                                #     print(f'RL error: {rl_error}, Obs error: {obs_error}, kf error: {kf_error}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'
                        avg_xyz_err, avg_xyz_or_err, avg_rl_llerr, avg_or_llerr = recording_results_ecef(data_truth_dic,[test_trajlist[0],
                                                   test_trajlist[-1]],tripIDlist_test, logdirname, baseline_mod, traj_record=False)
                        all_type_rl_3derr.append(avg_xyz_err)
                        all_type_or_3derr.append(avg_xyz_or_err)
                        all_type_rl_llerr.append(avg_rl_llerr)
                        all_type_or_llerr.append(avg_or_llerr)
                        print(f'More Test {testtype} finished.')

                record_difftype = pd.DataFrame(
                    {'Type': moreteststypelist, '3Derr': all_type_rl_3derr, 'or_3Derr':all_type_or_3derr,
                     'llerr': all_type_rl_llerr,'or_llerr': all_type_or_llerr})
                record_difftype.loc[len(record_difftype)] = ['average', np.mean(all_type_rl_3derr),np.mean(all_type_or_3derr),
                                                             np.mean(all_type_rl_llerr),np.mean(all_type_or_llerr)]
                record_difftype.to_csv(model_loggerdir + f'/record_diserr={np.mean(all_type_rl_3derr):.4}.csv',
                                       index=True)



