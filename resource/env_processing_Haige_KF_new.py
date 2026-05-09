import gym
from gym import spaces
import random
import numpy as np
import os
import numpy as np
import pandas as pd
from pathlib import Path
import glob as gl
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("../src")
from env.env_funcs_multic_Haige_hui import LOSPRRprocess
import gnss_lib.coordinates as coord
from funcs.timetransfer import *
import pymap3d.vincenty as pmv
from datetime import datetime, timedelta
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
from funcs.utilis import *
import folium
import pymap3d as pm
import time
import multiprocessing as mp
# from dataset.TD_rtkres2csv import *
import chardet


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
datatype = 'HaigeData'
base_directory='/mnt/sdb/home/tangjh/DYdata2024'    # /mnt/sdb/home/tangjh/DYdata2024
filesort='All'
obsHz_name=''
id_call=True
save_file = True # 保存处理数据
save_dop_file = True # 保存dop数据
process_gnss_features = True # 是否预处理gnss特征
record_dis = True # 是否记录误差统计
record_visualization = False # 记录可视化轨迹
load_data = False # 是否导入先前处理的数据，如果filefolders包含已有数据则会覆盖已有数据

err_max = 100
labelsize = 13

# Loop for each trip
if load_data:
    """
    输入对应数据地址，如果有些数据没有试试注释掉
    """
    with open(base_directory + '/env/raw_baseline_Haige_new_selfvel.pkl', "rb") as file:
        data_truth_dic = pickle.load(file)
    file.close()
    with open(base_directory + '/env/processed_features_BDS_Haige_new.pkl', "rb") as file:
        losfeature = pickle.load(file)
    file.close()
    with open(base_directory + '/env/raw_gnss_multic_Haige_new.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()
    with open(base_directory + '/env/raw_dop_Haige_new.pkl', "rb") as file:
        pdop_dic = pickle.load(file)
    file.close()
    satnum_df = pd.read_csv(f'{base_directory}/env/raw_satnum_BDS_Haige_new.csv')
    traj_sum_df = pd.read_csv(f'{base_directory}/env/raw_tripID_BDS_Haige_new.csv')
    sat_summary_multic_all = pd.read_csv(f'{base_directory}/env/raw_satnum_BDS_Haige_new.csv')
    tripIDlist = traj_sum_df['tripID'].values.tolist()
else:
    data_truth_dic={}
    pdop_dic = {}
    gnss_dic={}
    losfeature={}
    tripIDlist = []

truth_min_X=[]
truth_min_Y=[]
truth_min_Z=[]
truth_max_X=[]
truth_max_Y=[]
truth_max_Z=[]
kf_min_X=[]
kf_min_Y=[]
kf_min_Z=[]
kf_max_X=[]
kf_max_Y=[]
kf_max_Z=[]
truth_min_lat=[]
truth_min_lon=[]
truth_min_alt=[]
truth_max_lat=[]
truth_max_lon=[]
truth_max_alt=[]
kf_min_lat=[]
kf_min_lon=[]
kf_min_alt=[]
kf_max_lat=[]
kf_max_lon=[]
kf_max_alt=[]

dir_path=f'{base_directory}/dataset/{datatype}' # {base_directory}/envdata/{datatype}
# filefolders=os.listdir(dir_path)
# filefolders.sort()

"""
'20250318_1','20250325_1','20250325_2','20250407_1','20250407_2','20250513',
'20250620-AM_1','20250620-AM_2','20250620-PM_1','20250620-PM_2',
'20250701-AM','20250701-PM','20250703-AM_1','20250703-AM_2','20250703-PM_1','20250703-PM_2',
'20250708-AM_1','20250708-AM_2','20250708-PM_1','20250708-PM_2',
'20250711_1','20250711_2','20250718-AM_1','20250718-AM_2','20250718-PM_1','20250718-PM_2','20250719-AM_1','20250719-AM_2',
'20250723-AM_1','20250723-AM_2','20250723-PM_1','20250723-PM_2'
"""

filefolders = ['20250318_1','20250325_1','20250325_2','20250407_1','20250407_2','20250513',
'20250620-AM_1','20250620-AM_2','20250620-PM_1','20250620-PM_2',
'20250701-AM','20250701-PM','20250703-AM_1','20250703-AM_2','20250703-PM_1','20250703-PM_2',
'20250708-AM_1','20250708-AM_2','20250708-PM_1','20250708-PM_2',
'20250711_1','20250711_2','20250718-AM_1','20250718-AM_2','20250718-PM_1','20250718-PM_2','20250719-AM_1','20250719-AM_2',
'20250723-AM_1','20250723-AM_2','20250723-PM_1','20250723-PM_2']

for filefolder in filefolders:
    print(f'########## Processing data: {filefolder} #############')
    tripID=filefolder
    file_path=f'{dir_path}/{filefolder}'
    filelist=os.listdir(file_path)
    filelist.sort()
    resfile_list = []
    # path for data record
    plt_folder = f'{dir_path}/record_results_new'
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)
    for filename in filelist:
        """
        determine files for processing
        """
        if '100c' in filename or 'gga' in filename: # 参考真值
            NovAtelname = filename
        if 'obspvt_satVel' in filename: # gnss观测文件
            gnssname = filename
        if 'DAT' in filename and 'hexing' not in filename: # 速度文件
            velname = filename
        if 'ublox' in filename: # ublox gga file
            ubloxname = filename
        else:
            ubloxname = None
        if 'AI_GNSS' in filename:
            aignssname = filename
        else:
            aignssname = None


    # with open(reffile, 'rb') as f:
    #     rawdata = f.read()
    #     results = chardet.detect(rawdata)
    #     encoding = results['encoding']

    """
    read gnss data from Haige
    """
    gnss_path = f'{file_path}/{gnssname}'
    gnss_data = []
    spp_data = []
    max_len_gnss = 0
    max_len_rtk = 0
    with open(gnss_path, 'r', encoding='utf-8') as file:
        for line in file:
            if 'RTK Start' in line:
                split_line = [item.strip() for item in line.strip().split(':')]
                utc_epoch = split_line[1] # 提取utc时间

            if 'SATINFO' in line and 'GPST_SOW' not in line:
                # 使用逗号分隔符分割行，并去除换行符和多余空格
                split_line = [item.strip() for item in line.strip().split(',')]
                split_line[1] = utc_epoch
                gnss_data.append(split_line)
                if len(split_line) > max_len_gnss:
                    max_len_gnss = len(split_line)

            if 'USERPVT' in line and 'GPST_SOW' not in line:
                split_line = [item.strip() for item in line.strip().split(',')]
                split_line[1] = utc_epoch
                spp_data.append(split_line)
                if len(split_line) > max_len_rtk:
                    max_len_rtk = len(split_line)

    # 将所有行补齐，使其具有相同的列数，用NaN填充
    data_padded = [line + [np.nan] * (max_len_gnss - len(line)) for line in gnss_data]
    raw_gnss_pd = pd.DataFrame(data_padded)
    data_padded = [line + [np.nan] * (max_len_rtk - len(line)) for line in spp_data]
    raw_spp_pd = pd.DataFrame(data_padded)

    gnss_pd = raw_gnss_pd.iloc[:, [1,2,4,6,7,8,9,10,11,12,13,14,15]].astype(float)
    gnss_pd.columns = ['UTCtime','satID','pseudorange','dopple_m_s','CNR','EA','AA','SvPositionXEcefMeters',
              'SvPositionYEcefMeters','SvPositionZEcefMeters','SvVelocityXEcefMetersPerSecond','SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']
    # spp_pd = raw_spp_pd.iloc[:, [1,3,4,5]].astype(float)
    # spp_pd.columns = ['UTCtime','XEcefMeters_spp','YEcefMeters_spp','ZEcefMeters_spp']
    # # spp_pd = spp_pd[spp_pd['XEcefMeters_spp'] != 0] # 异常值剔除
    # columns_to_interpolate = ['XEcefMeters_spp','YEcefMeters_spp','ZEcefMeters_spp']
    # spp_pd[columns_to_interpolate] = spp_pd[columns_to_interpolate].replace(0, np.nan).interpolate(method='linear')

    """
    read ground truth data(GGA) from NovAtel(100c)
    """
    gtfile = f'{file_path}/{NovAtelname}'
    rawNovAtel_pd = pd.read_table(gtfile, sep=',', low_memory=False, header=None, encoding='utf-8')
    NovAtel_pd = rawNovAtel_pd.iloc[:, [1, 2, 4, 9]]
    NovAtel_pd.columns = ['UTCtime(hhmmss)', 'latitude(deg)', 'longitude(deg)', 'height(m)']
    NovAtel_pd = NovAtel_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
    NovAtel_pd[['latitude(deg)', 'longitude(deg)']] = NovAtel_pd[['latitude(deg)', 'longitude(deg)']].apply(
        lambda x: x // 100 + (x % 100) / 60)
    if  '20250325' in filefolder:
        # 数据采集，时间大于16点26分（UTC=082600）后的数据要作废
        NovAtel_pd = NovAtel_pd[NovAtel_pd['UTCtime(hhmmss)'] <= 082600.00]
    NovAtel_pd['UTCtime'] = NovAtel_pd['UTCtime(hhmmss)'].apply(lambda x: x//10000*3600+x%10000//100*60+x%100)
    NovAtel_pd.drop(columns=['UTCtime(hhmmss)'], inplace=True)

    """
    read velocity and position from DAT file
    * there are a few nan data in the speed_pd, but I don't plan to estimate them.
    """
    vel_path = f'{file_path}/{velname}'
    vel_data = []
    spp_data = []
    pdop_data = []
    std_data = []
    max_len = 0
    max_len_spp = 0
    max_len_pdop = 0
    max_len_std = 0
    outlier_velocity = 37 # m/s（国内最高限速33m/s）
    with open(vel_path, 'r', encoding='utf-8', errors='replace') as file:
        for idx,line in enumerate(file):
            if 'DHV' in line:
                split_line = [item.strip() for item in line.strip().split(',')]
                vel_data.append(split_line)
                if len(split_line) > max_len:
                    max_len = len(split_line)
            if 'GBGGA' in line:
                split_line = [item.strip() for item in line.strip().split(',')]
                spp_data.append(split_line)
                if len(split_line) > max_len_spp:
                    max_len_spp = len(split_line)
            if 'GNGGA' in line:
                split_line = [item.strip() for item in line.strip().split(',')]
                split_line = split_line[-15:]
                spp_data.append(split_line)
                if len(split_line) > max_len_spp:
                    max_len_spp = len(split_line)
                gga_idx = idx # gsa数据同个时间戳有重复，但一般只取gga后面的一条

            if ('GSA' in line) and (idx == gga_idx+1): # 提取PDOP信息，0318数据没有
                split_line = [item.strip() for item in line.strip().split(',')]
                pdop_data.append(split_line)
                if len(split_line) > max_len_pdop:
                    max_len_pdop = len(split_line)

            if 'GNRMC' in line:
                split_line = [item.strip() for item in line.strip().split(',')]
                utc_temp = [split_line[1]]
                std_idx = idx + 1

            try: # 只有7月份数据有std输出
                if ('POSSTD' in line) and (idx == std_idx):
                    split_line = [item.strip() for item in line.strip().split(',')]
                    std_data.append(utc_temp+split_line)
                    if len(split_line) > max_len_std:
                        max_len_std = len(split_line)
            except:
                pass

    # 提取速度数据
    colums_speed = ['UTCtime(hhmmss)', 'VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']
    index_list_speed = [1, 3, 4, 5]
    speed_pd = list2pd(vel_data,max_len,index_list_speed,colums_speed)
    speed_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
    speed_pd['UTCtime'] = speed_pd['UTCtime(hhmmss)'].apply(
        lambda x: x // 10000 * 3600 + x % 10000 // 100 * 60 + x % 100)
    colums_speed_reset = ['VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']
    speed_pd[colums_speed_reset] = speed_pd[colums_speed_reset] * (5/18) # km/h -> m/s
    speed_pd.drop(columns=['UTCtime(hhmmss)'], inplace=True)


    # 提取位置数据
    colums_spp = ['UTCtime(hhmmss)', 'latitude_spp_GGA', 'longitude_spp_GGA', 'Satnum', 'height_spp_GGA']
    index_list_spp = [1, 2, 4, 7, 9]
    spp_pd = list2pd(spp_data,max_len_spp,index_list_spp,colums_spp)
    spp_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
    satnum_pd = spp_pd['Satnum']
    spp_pd['UTCtime'] = spp_pd['UTCtime(hhmmss)'].apply(
        lambda x: x // 10000 * 3600 + x % 10000 // 100 * 60 + x % 100)
    spp_pd.drop(columns=['UTCtime(hhmmss)','Satnum'], inplace=True)
    spp_pd[['latitude_spp_GGA', 'longitude_spp_GGA']] = spp_pd[['latitude_spp_GGA', 'longitude_spp_GGA']].apply(
        lambda x: x // 100 + (x % 100) / 60)
    spp_llh = np.array([spp_pd['latitude_spp_GGA'], spp_pd['longitude_spp_GGA'], spp_pd['height_spp_GGA']])
    spp_llh2XYZ = np.array(
        [coord.geodetic2ecef(spp_llh[:, i]) for i in range(spp_llh.shape[1])])  # max(true_llh.shape)
    spp_pd['XEcefMeters_spp'] = spp_llh2XYZ[:, 0]
    spp_pd['YEcefMeters_spp'] = spp_llh2XYZ[:, 1]
    spp_pd['ZEcefMeters_spp'] = spp_llh2XYZ[:, 2]
    spp_xyz_pd = spp_pd[['XEcefMeters_spp','YEcefMeters_spp','ZEcefMeters_spp','UTCtime']]
    # smoothing data of position and velocity
    # position_smoothing_pd,speed_smoothing_pd = smooth_pos_vel(spp_xyz_pd.copy(),speed_pd.copy(), outlier_velocity)
    # speed_smoothing_pd = smooth_vel(speed_pd.copy(), outlier_velocity)
    nan_vel_num = speed_pd['VXEcefMeters_wls'].isna().sum()
    print(f'Nan epoch: {nan_vel_num}')

    # 提取pdop信息，0318没有pdop信息
    if len(pdop_data) > 0:
        colums_dop = ['PDOP', 'HDOP', 'VDOP']
        index_list_dop = [15, 16, 17]
        dop_pd = list2pd(pdop_data, max_len_pdop, index_list_dop, colums_dop)
        dop_pd['UTCtime'] = spp_pd['UTCtime']
        dop_pd['Satnum'] = satnum_pd

    # 提取spp的std信息
    if std_data:
        colums = ['UTCtime(hhmmss)', 'std_x', 'std_y','std_z']
        index_list = [0, 2, 3 ,4]
        std_pd = list2pd(std_data, max_len_std, index_list, colums)
        std_pd['UTCtime'] = spp_pd['UTCtime']

    """
    read ai gnss data（这里先不用）
    """
    if aignssname is not None:
        aignssfile = f'{file_path}/{aignssname}'
        aignss_data = []
        sppbl_data = []
        velbl_data = []
        max_len = 0
        max_len_bl = 0
        max_len_vel = 0
        with open(aignssfile, 'r', encoding='utf-8', errors='replace') as file:
            for idx,line in enumerate(file):
                if '$GDUT_RLAKF' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    aignss_data.append(split_line)
                    if len(split_line) > max_len:
                        max_len = len(split_line)
                if 'UserdPos1' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    sppbl_data.append(split_line)
                    if len(split_line) > max_len_bl:
                        max_len_bl = len(split_line)
                if 'UserdVel1' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    velbl_data.append(split_line)
                    if len(split_line) > max_len_vel:
                        max_len_vel = len(split_line)

        # 提取aignss数据
        colums = ['BDST(week)','BDST(sow)', 'XEcefMeters_AI', 'YEcefMeters_AI', 'ZEcefMeters_AI']
        index_list = [2, 3, 4, 5, 6]
        aignss_pd = list2pd(aignss_data, max_len, index_list, colums)
        aignss_pd.drop_duplicates(subset=['BDST(sow)']).astype(float).reset_index(drop=True)
        aignss_pd['UTCtime'] = aignss_pd['BDST(week)']*604800 + aignss_pd['BDST(sow)'] # BDST比UTC快18秒
        aignss_pd['UTCtime'] = aignss_pd['UTCtime'].apply(lambda x: x % 86400)-4
        aignss_pd.drop(columns=['BDST(sow)'], inplace=True)
        aignss_pd.drop(columns=['BDST(week)'], inplace=True)

        # 提取sppbl数据
        colums = ['BDST(week)','BDST(sow)', 'XEcefMeters_sppbl', 'YEcefMeters_sppbl', 'ZEcefMeters_sppbl']
        index_list = [1, 2, 3, 4, 5]
        sppbl_pd = list2pd(sppbl_data, max_len, index_list, colums)
        sppbl_pd.drop_duplicates(subset=['BDST(sow)']).astype(float).reset_index(drop=True)
        sppbl_pd['UTCtime'] = sppbl_pd['BDST(week)']*604800 + sppbl_pd['BDST(sow)'] # BDST比UTC快4秒,不计算闰秒
        sppbl_pd['UTCtime'] = sppbl_pd['UTCtime'].apply(lambda x: x % 86400)-4
        sppbl_pd.drop(columns=['BDST(sow)'], inplace=True)
        sppbl_pd.drop(columns=['BDST(week)'], inplace=True)

        # 提取velbl数据
        colums = ['BDST(week)','BDST(sow)', 'VXEcefMeters_wlsbl', 'VYEcefMeters_wlsbl', 'VZEcefMeters_wlsbl']
        index_list = [1, 2, 3, 4, 5]
        velbl_pd = list2pd(velbl_data, max_len, index_list, colums)
        velbl_pd.drop_duplicates(subset=['BDST(sow)']).astype(float).reset_index(drop=True)
        velbl_pd['UTCtime'] = velbl_pd['BDST(week)']*604800 + velbl_pd['BDST(sow)'] # BDST比UTC快4秒,不计算闰秒
        velbl_pd['UTCtime'] = velbl_pd['UTCtime'].apply(lambda x: x % 86400)-4
        velbl_pd.drop(columns=['BDST(sow)'], inplace=True)
        velbl_pd.drop(columns=['BDST(week)'], inplace=True)

    """
    read ublox data (gga)（这里先不用）
    """
    if ubloxname is not None:
        ubloxfile = f'{file_path}/{ubloxname}'
        ublox_data = []
        max_len = 0
        with open(ubloxfile, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                if 'GPGGA' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    ublox_data.append(split_line)
                    if len(split_line) > max_len:
                        max_len = len(split_line)
        colums_ublox = ['UTCtime(hhmmss)', 'latitude_ublox_GGA', 'longitude_ublox_GGA', 'height_ublox_GGA']
        index_list_ublox = [1, 2, 4, 9]
        ublox_pd = list2pd(ublox_data, max_len, index_list_ublox, colums_ublox)
        ublox_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
        ublox_pd['UTCtime'] = ublox_pd['UTCtime(hhmmss)'].apply(
            lambda x: x // 10000 * 3600 + x % 10000 // 100 * 60 + x % 100)
        ublox_pd.drop(columns=['UTCtime(hhmmss)'], inplace=True)
        ublox_pd[['latitude_ublox_GGA', 'longitude_ublox_GGA']] = ublox_pd[['latitude_ublox_GGA', 'longitude_ublox_GGA']].apply(
            lambda x: x // 100 + (x % 100) / 60)
        ublox_llh = np.array([ublox_pd['latitude_ublox_GGA'], ublox_pd['longitude_ublox_GGA'], ublox_pd['height_ublox_GGA']])
        ublox_llh2XYZ = np.array(
            [coord.geodetic2ecef(ublox_llh[:, i]) for i in range(ublox_llh.shape[1])])  # max(true_llh.shape)
        ublox_pd['XEcefMeters_ublox'] = ublox_llh2XYZ[:, 0]
        ublox_pd['YEcefMeters_ublox'] = ublox_llh2XYZ[:, 1]
        ublox_pd['ZEcefMeters_ublox'] = ublox_llh2XYZ[:, 2]
        # smoothing data of position and velocity
        # ublox_xyz_pd = ublox_pd[['XEcefMeters_ublox', 'YEcefMeters_ublox', 'ZEcefMeters_ublox', 'UTCtime']]
        # ublox_smoothing_pd, _ = smooth_pos_vel(ublox_xyz_pd.copy(), speed_pd.copy(), outlier_velocity)
        # nan_vel_num = ublox_smoothing_pd['XEcefMeters_ublox'].isna().sum()
        # print(f'Nan epoch: {nan_vel_num}')

    """
    Calculate velocity by self
    注意：cov_v是计算速度的协方差，看用来代替Q
    """
    speed_self_pd,_,cov_v = WLS_velocity_solution(tripID,gnss_pd,spp_pd) # speed_or_pd

    """
    Time processing merge
    """
    merge_columns = ['UTCtime']
    # baseline_alltime = rtk_pd.merge(NovAtel_pd, on=merge_columns, suffixes=('_rtk', ''))
    # baseline_alltime = baseline_alltime.merge(spp_pd, on=merge_columns, suffixes=('_spp', ''))
    baseline_alltime = spp_pd.merge(NovAtel_pd, on = merge_columns, suffixes = ('', ''))
    baseline_alltime = baseline_alltime.merge(speed_pd, on = merge_columns, suffixes = ('', ''))
    baseline_alltime = baseline_alltime.merge(speed_self_pd, on = merge_columns, suffixes = ('', ''))
    # baseline_alltime = baseline_alltime.merge(position_smoothing_pd, on = merge_columns, suffixes = ('', '_smoothing'))
    if ubloxname is not None:
        baseline_alltime = baseline_alltime.merge(ublox_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.dropna(subset=['XEcefMeters_ublox']).reset_index(drop=True)  # 剔除无定位数据的行
    if std_data:
        baseline_alltime = baseline_alltime.merge(std_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.dropna(subset=['std_x']).reset_index(drop=True)  # 剔除无定位数据的行
    # if aignssname is not None:
    #     baseline_alltime = baseline_alltime.merge(aignss_pd, on=merge_columns, suffixes=('', ''))
    #     baseline_alltime = baseline_alltime.merge(sppbl_pd, on=merge_columns, suffixes=('', ''))
    #     baseline_alltime = baseline_alltime.merge(velbl_pd, on=merge_columns, suffixes=('', ''))
    #     baseline_alltime = baseline_alltime.dropna(subset=['XEcefMeters_AI']).reset_index(drop=True)  # 剔除无定位数据的行
    # baseline_alltime = baseline_alltime.merge(speed_smoothing_pd, on = merge_columns, suffixes = ('', '_smoothing'))
    baseline = baseline_alltime.drop_duplicates(subset=['UTCtime']).reset_index(drop=True) # 剔除重复行，如果重复就只提取第一行
    baseline = baseline.dropna(subset=['XEcefMeters_spp']).reset_index(drop=True)  # 剔除无定位数据的行
    baseline = baseline.dropna(subset=['VXEcefMeters_self']).reset_index(drop=True)  # 剔除无速度数据的行

    """
    处理原生速度异常值，可以选择是否处理
    """
    colums_v = ['VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']
    colums_spp = ['XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']
    v_wls = baseline.loc[:,colums_v].values
    for i in range(len(baseline)):
        x0 = baseline.loc[i, colums_spp].values
        x_llh = np.array(pm.ecef2geodetic(x0[0], x0[1], x0[2])).T
        v_enu = np.array(pm.ecef2enuv(v_wls[i,0], v_wls[i,1], v_wls[i,2], x_llh[0], x_llh[1])).T
        if np.abs(v_enu[2]) > v_up_th:
            v_wls[i, :] = (v_wls[i - 1, :] + v_wls[i - 2, :]) * 0.5
            baseline.loc[i,colums_v] = v_wls[i, :]

    """
    record distance error of data
    """
    true_llh = np.array([baseline['latitude(deg)'], baseline['longitude(deg)'],baseline['height(m)']])
    true_llh2XYZ = np.array(
        [coord.geodetic2ecef(true_llh[:, i]) for i in range(true_llh.shape[1])])  # max(true_llh.shape)
    baseline['ecefX_gt'] = true_llh2XYZ[:,0]
    baseline['ecefY_gt'] = true_llh2XYZ[:,1]
    baseline['ecefZ_gt'] = true_llh2XYZ[:,2]

    """
    estimate velocity with NovAtel
    """
    speed_gt = np.vstack((np.diff(true_llh2XYZ, axis=0),np.zeros(3)))
    speed_spp = baseline[['VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']].to_numpy()
    speed_self_spp = baseline[['VXEcefMeters_self','VYEcefMeters_self','VZEcefMeters_self']].to_numpy()
    # speed_spp_smoothing = baseline[['VXEcefMeters_wls_smoothing','VYEcefMeters_wls_smoothing','VZEcefMeters_wls_smoothing']].to_numpy()

    """
    Solve with Kalman filter and Saga-husa AKF
    """
    sigma_v = 0.6
    sigma_x = 12.0 # 20
    initial_P = 0.3
    zs = baseline[['XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']].to_numpy()
    us = baseline[['VXEcefMeters_wls', 'VYEcefMeters_wls', 'VZEcefMeters_wls']].to_numpy()
    us_self = baseline[['VXEcefMeters_self', 'VYEcefMeters_self', 'VZEcefMeters_self']].to_numpy()

    baseline[['XEcefMeters_kf','YEcefMeters_kf','ZEcefMeters_kf']] = Kalmanfilter(zs, us, baseline, sigma_v, sigma_x, initial_P, outlier_velocity)

    baseline[['XEcefMeters_kf_self', 'YEcefMeters_kf_self', 'ZEcefMeters_kf_self']] = Kalmanfilter(zs, us_self, baseline, sigma_v / 2, sigma_x,initial_P, outlier_velocity)

    # baseline[['XEcefMeters_sagakf','YEcefMeters_sagakf','ZEcefMeters_sagakf']] = Saga_husa_AKF(zs, us, baseline, sigma_v, sigma_x, initial_P, outlier_velocity,lamb = 0.9)

    """
    坐标系转换
    """
    # 计算大地坐标系的结果
    guess_xyz_spp = np.array([baseline['XEcefMeters_spp'], baseline['YEcefMeters_spp'], baseline['ZEcefMeters_spp']])
    guess_spp_XYZ2llh = np.array(
        [coord.ecef2geodetic(guess_xyz_spp[:, i]) for i in range(guess_xyz_spp.shape[1])])
    guess_xyz_kf = np.array([baseline['XEcefMeters_kf'], baseline['YEcefMeters_kf'], baseline['ZEcefMeters_kf']])
    guess_kf_XYZ2llh = np.array(
        [coord.ecef2geodetic(guess_xyz_kf[:, i]) for i in range(guess_xyz_kf.shape[1])])
    guess_xyz_kf_self = np.array([baseline['XEcefMeters_kf_self'], baseline['YEcefMeters_kf_self'], baseline['ZEcefMeters_kf_self']])
    guess_kf_self_XYZ2llh = np.array(
        [coord.ecef2geodetic(guess_xyz_kf_self[:, i]) for i in range(guess_xyz_kf_self.shape[1])])
    # if aignssname is not None:
    #     guess_xyz_aignss = np.array([baseline['XEcefMeters_AI'], baseline['YEcefMeters_AI'], baseline['ZEcefMeters_AI']])
    #     guess_aignss_XYZ2llh = np.array(
    #         [coord.ecef2geodetic(guess_xyz_aignss[:, i]) for i in range(guess_xyz_aignss.shape[1])])
    #     guess_xyz_sppbl = np.array([baseline['XEcefMeters_sppbl'], baseline['YEcefMeters_sppbl'], baseline['ZEcefMeters_sppbl']])
    #     guess_sppbl_XYZ2llh = np.array(
    #         [coord.ecef2geodetic(guess_xyz_sppbl[:, i]) for i in range(guess_xyz_sppbl.shape[1])])

    # 计算ENU坐标系的结果（验证一下算水平误差的区别，可以不用看）
    guess_NED_spp, truth_NED_gt, guess_NED_kf = [],[],[]
    ref_local = coord.LocalCoord.from_ecef(guess_xyz_spp[:, 0])
    for i in range(guess_xyz_spp.shape[1]):
        guess_NED_spp.append(ref_local.ecef2ned(guess_xyz_spp[:,i,None])[:, 0])
        guess_NED_kf.append(ref_local.ecef2ned(guess_xyz_kf[:,i,None])[:, 0])
        truth_NED_gt.append(ref_local.ecef2ned(true_llh2XYZ.T[:,i,None])[:, 0])
    guess_NED_spp = np.array(guess_NED_spp)
    guess_NED_kf = np.array(guess_NED_kf)
    truth_NED_gt = np.array(truth_NED_gt)

    # baseline['latitude(deg)_spp'] = guess_spp_XYZ2llh[:,0]
    # baseline['longitude(deg)_spp'] = guess_spp_XYZ2llh[:,1]
    # baseline['height(m)_spp'] = guess_spp_XYZ2llh[:,2]

    """
    统计误差结果计算
    """
    dist_err_xyz_spp = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_spp)
    dist_err_xyz_kf = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_kf)
    dist_err_xyz_kf_self = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_kf_self)
    dist_ll_spp = vincenty_distance(guess_spp_XYZ2llh, true_llh.T)
    dist_ll_kf = vincenty_distance(guess_kf_XYZ2llh, true_llh.T)
    dist_ll_kf_self = vincenty_distance(guess_kf_self_XYZ2llh, true_llh.T)
    dist_h_spp = np.abs(guess_spp_XYZ2llh.T[2, :] - true_llh[2, :])
    dist_h_kf = np.abs(guess_kf_XYZ2llh.T[2, :] - true_llh[2, :])
    dist_h_kf_self = np.abs(guess_kf_self_XYZ2llh.T[2, :] - true_llh[2, :])
    dist_ned_2D_spp, dist_ned_h_spp = dist_err_NED(truth_NED_gt,guess_NED_spp)
    dist_ned_2D_kf, dist_ned_h_kf = dist_err_NED(truth_NED_gt,guess_NED_kf)
    speed_err_xyz = dist_err_XYZ(speed_spp.T, speed_gt.T)
    speed_self_err_xyz = dist_err_XYZ(speed_self_spp.T, speed_gt.T)
    # if aignssname is not None:
    #     dist_err_xyz_aignss = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_aignss)
    #     dist_err_xyz_sppbl = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_sppbl)
    #     dist_ll_aignss = vincenty_distance(guess_aignss_XYZ2llh, true_llh.T)
    #     dist_ll_sppbl = vincenty_distance(guess_sppbl_XYZ2llh, true_llh.T)
    #     dist_h_aignss = np.abs(guess_aignss_XYZ2llh.T[2, :] - true_llh[2, :])
    #     dist_h_sppbl = np.abs(guess_sppbl_XYZ2llh.T[2, :] - true_llh[2, :])
    # speed_err_xyz_smoothing = dist_err_XYZ(speed_spp_smoothing.T, speed_gt.T)
    # 剔除异常值
    err_idx_1 = np.where(dist_err_xyz_spp > 1e4)
    err_idx_2 = np.where(dist_err_xyz_kf > 1e4)
    dist_err_xyz_spp = dist_err_xyz_spp[dist_err_xyz_spp <= 1e5]
    dist_err_xyz_kf = dist_err_xyz_kf[dist_err_xyz_kf <= 1e5]
    dist_err_xyz_kf_self = dist_err_xyz_kf_self[dist_err_xyz_kf_self <= 1e5]
    dist_ned_2D_spp = dist_ned_2D_spp[dist_ned_2D_spp <= 1e5]
    dist_ned_2D_kf = dist_ned_2D_kf[dist_ned_2D_kf <= 1e5]
    dist_ll_kf_self = dist_ll_kf_self[dist_ll_kf_self <= 1e5]
    dist_ned_h_spp = dist_ned_h_spp[dist_ned_h_spp <= 1e5]
    dist_ned_h_kf = dist_ned_h_kf[dist_ned_h_kf <= 1e5]
    dist_h_spp = dist_h_spp[dist_h_spp <= 1e5]
    dist_h_kf = dist_h_kf[dist_h_kf <= 1e5]
    dist_h_kf_self = dist_h_kf_self[dist_h_kf_self <= 1e5]
    baseline = baseline.drop(baseline.index[err_idx_1]) # 数据中剔除异常的列
    baseline = baseline.drop(baseline.index[err_idx_2]).reset_index(drop=True)  # 数据中剔除异常的列
    print(f'Outlier num: {len(err_idx_2)+len(err_idx_1)}')

    if record_dis:
        plt.figure()
        plot_dist(dist_err_xyz_spp, err_max, 'SPP')
        plot_dist(dist_err_xyz_kf,err_max,'KF')
        plot_dist(dist_err_xyz_kf_self, err_max, 'KF-v2')
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Distance error distribution of ecef in {tripID}', fontsize=labelsize)
        plt.xlabel('Distance error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Haige_ecef_disterr_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(dist_err_xyz_spp, label='SPP, average error={:.3}m'.format(np.nanmean(dist_err_xyz_spp)))
        plt.plot(dist_err_xyz_kf, label='KF, average error={:.3}m'.format(np.nanmean(dist_err_xyz_kf)))
        plt.plot(dist_err_xyz_kf_self, label='KF-v2, average error={:.3}m'.format(np.nanmean(dist_err_xyz_kf_self)))
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Distance error of ecef in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Distance error (m)', fontsize=labelsize)
        #plt.ylim([0, 100])
        plt.savefig(f'{plt_folder}/Traj_100_ecef_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plot_dist(dist_ned_2D_spp, err_max, 'SPP')
        plot_dist(dist_ned_2D_kf, err_max, 'KF')
        plot_dist(dist_ll_kf_self, err_max, 'KF-v2')
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Horizontal error distribution in {tripID}', fontsize=labelsize)
        plt.xlabel('Horizontal error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Haige_ll_disterrdistr_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(dist_ned_2D_spp, label='SPP, average error={:.3}m'.format(np.nanmean(dist_ned_2D_spp)))
        plt.plot(dist_ned_2D_kf, label='KF, average error={:.3}m'.format(np.nanmean(dist_ned_2D_kf)))
        plt.plot(dist_ll_kf_self, label='KF-v2, average error={:.3}m'.format(np.nanmean(dist_ll_kf_self)))
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Horizontal error in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Horizontal error (m)', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Traj_org_ll_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plot_dist(dist_ned_h_spp, err_max, 'SPP')
        plot_dist(dist_ned_h_kf, err_max, 'KF')
        plot_dist(dist_h_kf_self, err_max, 'KF-v2')
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Elevation error distribution in {tripID}', fontsize=labelsize)
        plt.xlabel('Elevation error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Haige_h_disterrdistr_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(dist_h_spp, label=f'SPP')
        plt.plot(dist_h_kf, label=f'KF')
        plt.plot(dist_h_kf_self, label=f'KF-v2')
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Elevation error in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Elevation error (m)', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Traj_org_h_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(speed_err_xyz, label=f'Velocity: error={np.nanmean(speed_err_xyz):.2}+{np.nanstd(speed_err_xyz):.2}m/s')
        plt.plot(speed_self_err_xyz, label=f'Velocity self: error={np.nanmean(speed_self_err_xyz):.2}+{np.nanstd(speed_self_err_xyz):.2}m/s')
        # plt.plot(speed_err_xyz_smoothing, label=f'Velocity-smoothing: error={np.nanmean(speed_err_xyz_smoothing):.2}+{np.nanstd(speed_err_xyz_smoothing):.2}m/s')
        plt.grid()
        plt.legend(fontsize=labelsize)
        # plt.ylim([0, 20])
        plt.title(f'Velocity error in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Velocity error (m/s)', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Traj_org_Velocity_{tripID}.png', bbox_inches='tight')
        plt.show()
        # plt.figure()
        # plt.plot(HDOP_rtk)
        # plt.grid()
        # plt.title(f'HDOP of RTK in {tripID}', fontsize=labelsize)
        # plt.xlabel('Step', fontsize=labelsize)
        # plt.ylabel('HDOP', fontsize=labelsize)
        # plt.ylim([0, 10])
        # plt.savefig(f'{plt_folder}/Traj_HDOP_{tripID}.png', bbox_inches='tight')
        # plt.show()
        #
        # plt.figure()
        # plt.plot(Satnum)
        # plt.grid()
        # plt.title(f'Satellite number of RTK in {tripID}', fontsize=labelsize)
        # plt.xlabel('Step', fontsize=labelsize)
        # plt.ylabel('Satellite number', fontsize=labelsize)
        # plt.savefig(f'{plt_folder}/Traj_Satnum_{tripID}.png', bbox_inches='tight')
        # plt.show()
        if len(pdop_data) > 0:
            pdop = dop_pd['PDOP'].values
            satnum = dop_pd['Satnum'].values

            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(pdop, 'b-', label='PDOP')
            ax1.set_xlabel('Epoch (s)', fontsize=labelsize+3)
            ax1.set_ylabel('PDOP', color='b',fontsize=labelsize+3)
            ax1.spines['left'].set_color('b')
            ax1.tick_params(axis='y', labelcolor='b',labelsize=labelsize)
            ax1.tick_params(axis='x', labelsize=labelsize)
            ax1.set_ylim(0, 20)

            ax2 = ax1.twinx()
            ax2.plot(satnum, 'r--', label='Satellite number')
            ax2.set_ylabel('Satellite number', color='r', fontsize=labelsize+3)
            ax2.spines['right'].set_color('r')
            ax2.tick_params(axis='y', labelcolor='r',labelsize=labelsize)
            # fig.legend(fontsize=labelsize)
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            plt.savefig(f'{plt_folder}/PDOP_Satnum_{tripID}.png', bbox_inches='tight')
            plt.show()


    # if position_outlier_smoothing:
    #     colums_name = ['XEcefMeters_spp_smooth','YEcefMeters_spp_smooth','ZEcefMeters_spp_smooth']
    #     baseline[colums_name]=baseline[['XEcefMeters_spp','YEcefMeters_spp','ZEcefMeters_spp']]
    #     dist_err_xyz_spp = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_spp)
    #     outlier_error = 500
    #     outlier_idx = np.where(dist_err_xyz_spp > outlier_error)[0]
    #     print(f'outlier error: {outlier_error}, num: {len(outlier_idx)}')
    #     baseline.loc[outlier_idx,colums_name] = np.nan
    #     baseline.interpolate(method='linear', inplace=True)

    if record_visualization:
        truth_gt = baseline[['latitude(deg)', 'longitude(deg)']].to_numpy()
        spp_pd = baseline[['latitude_spp_GGA', 'longitude_spp_GGA']].to_numpy()
        kf_pd = guess_kf_XYZ2llh
        UTC_time = baseline['UTCtime'].to_numpy()
        llh_gt_500 = [truth_gt[-20, 0], truth_gt[-20, 1]]
        google_satellite = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        m = folium.Map(location=llh_gt_500,tiles=google_satellite, attr='Google', zoom_start=18, zoom_max =25)
        for index in range(len(truth_gt)):  # len(pd_train)-50,
            if index % 3 == 0:
                utc = UTC_time[index]
                folium.Circle(radius=0.2, location=[truth_gt[index, 0], truth_gt[index, 1]], popup=f'Ground Truth:{index},utc:{utc}',
                              color='yellow', fill=False).add_to(m)
                folium.Circle(radius=0.2, location=[spp_pd[index,0],spp_pd[index,1]],popup=f'SPP:{index},utc:{utc}',
                              color='cyan',fill=False).add_to(m)
                # folium.Circle(radius=0.2, location=[kf_pd[index,0],spp_pd[index,1]],popup=f'KF:{index},utc:{utc}',
                #               color='red',fill=False).add_to(m)
        m.save(f"{plt_folder}/{tripID}.html")

    # to record the RTK ambiguity resolution
    # rtk_ambiguity_ratio = np.array(baseline_pd['ratio'])
    # print(f'{filefolder}={np.mean(rtk_ambiguity_ratio)}')
    tripIDlist.append(tripID)
    print(f'{tripID}, rtkresidual num, {len(baseline)}, satpos num, final gnss num, {len(gnss_pd)}')

    if process_gnss_features:
        LOSPRR = LOSPRRprocess(gnss_pd, NovAtel_pd, tripID, dir_path, baseline)
        featureall, sat_summary_multic, interrupt_dic = LOSPRR.getitemECEF_rtkres(gnss_pd, id_call)
        # record interrupt timestep
        # interrupt_pd = pd.DataFrame(interrupt_dic)
        # interrupt_pd.to_csv(file_path + f'interrupt_{tripID}_Haige_record.csv', index=True)
        # extract time step of baseline in the gnss data
        feature_steps = list(featureall.keys())
        baseline = baseline[baseline['UTCtime'].isin(feature_steps)].reset_index(drop=True)
        losfeature[tripID] = featureall
        gnss_dic[tripID] = gnss_pd
        try:
            sat_summary_multic_all.loc[:, tripID] = pd.DataFrame({tripID: sat_summary_multic['Nums']})
            # sat_summary_multic_all = pd.concat([sat_summary_multic_all, pd.DataFrame({tripID:sat_summary_multic['Nums']})], sort=False)
        except:
            sat_summary_multic_all = sat_summary_multic.rename(columns={'Nums': tripID}) # 统计所有轨迹数据的卫星数、最大值等

    data_truth_dic[tripID] = baseline

    if len(pdop_data) > 0:
        pdop_dic[tripID] = dop_pd

    # with open(base_directory + f'/env/processed_features_multic_lla_cid_{tripID}.pkl', 'wb') as value_file:
    #     pickle.dump(featureall, value_file, True)
    # value_file.close()
    # with open(base_directory + f'/env/raw_baseline_{tripID}.pkl', 'wb') as value_file:
    #     pickle.dump(baseline, value_file, True)
    # value_file.close()
    # with open(base_directory + f'/env/raw_gnss_multic_{tripID}.pkl', 'wb') as value_file:
    #     pickle.dump(gnss_pd, value_file, True)
    # value_file.close()
    # sat_summary_multic_all.to_csv(base_directory + f'/env/raw_satnum_multic_lla_50_{tripID}.csv', index=True)

    # truth XYZ min max baselin XYZ min max
    truth_min_X.append(np.min(baseline['ecefX_gt'].to_numpy()))
    truth_min_Y.append(np.min(baseline['ecefY_gt'].to_numpy()))
    truth_min_Z.append(np.min(baseline['ecefZ_gt'].to_numpy()))
    truth_max_X.append(np.max(baseline['ecefX_gt'].to_numpy()))
    truth_max_Y.append(np.max(baseline['ecefY_gt'].to_numpy()))
    truth_max_Z.append(np.max(baseline['ecefZ_gt'].to_numpy()))

    kf_min_X.append(np.min(baseline['XEcefMeters_spp'].to_numpy()))
    kf_min_Y.append(np.min(baseline['YEcefMeters_spp'].to_numpy()))
    kf_min_Z.append(np.min(baseline['ZEcefMeters_spp'].to_numpy()))
    kf_max_X.append(np.max(baseline['XEcefMeters_spp'].to_numpy()))
    kf_max_Y.append(np.max(baseline['YEcefMeters_spp'].to_numpy()))
    kf_max_Z.append(np.max(baseline['ZEcefMeters_spp'].to_numpy()))

    truth_min_lat.append(np.min(baseline['latitude(deg)'].to_numpy()))
    truth_min_lon.append(np.min(baseline['longitude(deg)'].to_numpy()))
    truth_min_alt.append(np.min(baseline['height(m)'].to_numpy()))
    truth_max_lat.append(np.max(baseline['latitude(deg)'].to_numpy()))
    truth_max_lon.append(np.max(baseline['longitude(deg)'].to_numpy()))
    truth_max_alt.append(np.max(baseline['height(m)'].to_numpy()))

    kf_min_lat.append(np.min(baseline['latitude_spp_GGA'].to_numpy()))
    kf_min_lon.append(np.min(baseline['longitude_spp_GGA'].to_numpy()))
    kf_min_alt.append(np.min(baseline['height_spp_GGA'].to_numpy()))
    kf_max_lat.append(np.max(baseline['latitude_spp_GGA'].to_numpy()))
    kf_max_lon.append(np.max(baseline['longitude_spp_GGA'].to_numpy()))
    kf_max_alt.append(np.max(baseline['height_spp_GGA'].to_numpy()))

if save_file:
    with open(base_directory + f'/env/processed_features_BDS_Haige_new_selfvel.pkl', 'wb') as value_file:
        pickle.dump(losfeature, value_file, True)
    value_file.close()
    with open(base_directory + f'/env/raw_baseline_Haige_new_selfvel.pkl', 'wb') as value_file:
        pickle.dump(data_truth_dic, value_file, True)
    value_file.close()
    # with open(base_directory + f'/env/raw_gnss_multic_Haige_new.pkl', 'wb') as value_file:
    #     pickle.dump(gnss_dic, value_file, True)
    # value_file.close()
    sat_summary_multic_all.to_csv(base_directory + f'/env/raw_satnum_BDS_Haige_new.csv', index=True)

    tripID_df=pd.DataFrame(tripIDlist, columns=['tripID'])
    tripID_df['ecefX_min']=truth_min_X
    tripID_df['ecefY_min']=truth_min_Y
    tripID_df['ecefZ_min']=truth_min_Z
    tripID_df['ecefX_max']=truth_max_X
    tripID_df['ecefY_max']=truth_max_Y
    tripID_df['ecefZ_max']=truth_max_Z

    tripID_df['ecefX_min_kf']=kf_min_X
    tripID_df['ecefY_min_kf']=kf_min_Y
    tripID_df['ecefZ_min_kf']=kf_min_Z
    tripID_df['ecefX_max_kf']=kf_max_X
    tripID_df['ecefY_max_kf']=kf_max_Y
    tripID_df['ecefZ_max_kf']=kf_max_Z

    tripID_df['lat_min']=truth_min_lat
    tripID_df['lon_min']=truth_min_lon
    tripID_df['alt_min']=truth_min_alt
    tripID_df['lat_max']=truth_max_lat
    tripID_df['lon_max']=truth_max_lon
    tripID_df['alt_max']=truth_max_alt

    tripID_df['lat_min_kf']=kf_min_lat
    tripID_df['lon_min_kf']=kf_min_lon
    tripID_df['alt_min_kf']=kf_min_alt
    tripID_df['lat_max_kf']=kf_max_lat
    tripID_df['lon_max_kf']=kf_max_lon
    tripID_df['alt_max_kf']=kf_max_alt

    def convert_to_decimal(deg):
        return int(deg / 100) + (deg % 100) / 60

    tripID_df.to_csv(base_directory + f'/env/raw_tripID_BDS_Haige_new.csv', index=True)

if save_dop_file:
    with open(base_directory + f'/env/raw_dop_Haige_new.pkl', 'wb') as value_file:
        pickle.dump(pdop_dic, value_file, True)
    value_file.close()

print('Processing finish !')