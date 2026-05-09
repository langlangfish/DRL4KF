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
from data_params import *
import pymap3d as pm
import time
import multiprocessing as mp
# from dataset.TD_rtkres2csv import *
import chardet


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datatype = 'HaigeData'
base_directory='/mnt/sdb/home/tangjh/DYdata2024'    # /mnt/sdb/home/tangjh/DYdata2024
filesort='All'
obsHz_name=''
id_call=True
save_file = False # 保存处理数据
save_dop_file = True # 保存dop数据
record_dis = False # 是否记录误差统计
record_visualization = False # 记录可视化轨迹
env_split = True # 环境划分统计
plot_split_dis = True # 是否分场景画图

err_max = 100
labelsize = 13

# Loop for each trip
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
'20250711_1','20250711_2',
'20250718-AM_1','20250718-AM_2','20250718-PM_1','20250718-PM_2','20250719-AM_1','20250719-AM_2'
"""

filefolders = ['20250708-PM_1','20250711_1','20250718-AM_1','20250718-PM_1',]
methodlist = ['spp','sppbl', 'kf', 'AI']  # 'spp','sppbl','kf','ublox','hexing','AI'
"""
'20250708-AM_1','20250708-PM_1'没ublox,
'20250718-AM_1','20250718-AM_2','20250718-PM_1','20250718-PM_2','20250719-AM_1' 有hexing和ublox
'20250711_1' 没ublox，AI异常bug多
"""
def statistic_data(data):
    mean = np.mean(data)
    std = np.std(data)
    CEF = np.percentile(np.sort(data), 50)
    one_sigma = np.percentile(np.sort(data), 67)
    two_sigma = np.percentile(np.sort(data), 95)
    max = np.max(data)
    min = np.min(data)
    return [mean,std,one_sigma,two_sigma,max,min,CEF]

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
    aignssname = None
    ubloxname = None
    hexingname = None
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
        if 'ubx' in filename: # ublox gga file
            ubloxname = filename
        if 'hexing' in filename or 'Hexing' in filename:
            hexingname = filename
        if 'AI_GNSS' in filename:
            aignssname = filename

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
    max_len = 0
    max_len_spp = 0
    max_len_pdop = 0
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

    """
    read ai gnss data
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
        # aignss_pd.drop(columns=['BDST(sow)'], inplace=True)
        # aignss_pd.drop(columns=['BDST(week)'], inplace=True)

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
    read ublox data (gga)
    """
    if ubloxname is not None:
        ubloxfile = f'{file_path}/{ubloxname}'
        ublox_data = []
        max_len = 0
        with open(ubloxfile, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                if 'GNGGA' in line:
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
    read hexing data (gga)
    """
    if hexingname is not None:
        hexingfile = f'{file_path}/{hexingname}'
        hexing_data = []
        max_len = 0
        with open(hexingfile, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                if 'GNGGA' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    hexing_data.append(split_line)
                    if len(split_line) > max_len:
                        max_len = len(split_line)
        colums_hexing = ['UTCtime(hhmmss)', 'latitude_hexing_GGA', 'longitude_hexing_GGA', 'height_hexing_GGA']
        index_list_hexing = [1, 2, 4, 9]
        hexing_pd = list2pd(hexing_data, max_len, index_list_hexing, colums_hexing)
        hexing_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
        hexing_pd['UTCtime'] = hexing_pd['UTCtime(hhmmss)'].apply(
            lambda x: x // 10000 * 3600 + x % 10000 // 100 * 60 + x % 100)
        hexing_pd.drop(columns=['UTCtime(hhmmss)'], inplace=True)
        hexing_pd[['latitude_hexing_GGA', 'longitude_hexing_GGA']] = hexing_pd[['latitude_hexing_GGA', 'longitude_hexing_GGA']].apply(
            lambda x: x // 100 + (x % 100) / 60)
        hexing_llh = np.array([hexing_pd['latitude_hexing_GGA'], hexing_pd['longitude_hexing_GGA'], hexing_pd['height_hexing_GGA']])
        hexing_llh2XYZ = np.array(
            [coord.geodetic2ecef(hexing_llh[:, i]) for i in range(hexing_llh.shape[1])])  # max(true_llh.shape)
        hexing_pd['XEcefMeters_hexing'] = hexing_llh2XYZ[:, 0]
        hexing_pd['YEcefMeters_hexing'] = hexing_llh2XYZ[:, 1]
        hexing_pd['ZEcefMeters_hexing'] = hexing_llh2XYZ[:, 2]
        # smoothing data of position and velocity
        # ublox_xyz_pd = ublox_pd[['XEcefMeters_ublox', 'YEcefMeters_ublox', 'ZEcefMeters_ublox', 'UTCtime']]
        # ublox_smoothing_pd, _ = smooth_pos_vel(ublox_xyz_pd.copy(), speed_pd.copy(), outlier_velocity)
        # nan_vel_num = ublox_smoothing_pd['XEcefMeters_ublox'].isna().sum()
        # print(f'Nan epoch: {nan_vel_num}')

    """
    Time processing merge
    """
    merge_columns = ['UTCtime']
    # baseline_alltime = rtk_pd.merge(NovAtel_pd, on=merge_columns, suffixes=('_rtk', ''))
    # baseline_alltime = baseline_alltime.merge(spp_pd, on=merge_columns, suffixes=('_spp', ''))
    baseline_alltime = spp_pd.merge(NovAtel_pd, on = merge_columns, suffixes = ('', ''))
    baseline_alltime = baseline_alltime.merge(speed_pd, on = merge_columns, suffixes = ('', ''))
    # baseline_alltime = baseline_alltime.merge(position_smoothing_pd, on = merge_columns, suffixes = ('', '_smoothing'))
    if 'ublox' in methodlist:
        baseline_alltime = baseline_alltime.merge(ublox_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.dropna(subset=['XEcefMeters_ublox']).reset_index(drop=True)  # 剔除无定位数据的行
    if 'hexing' in methodlist:
        baseline_alltime = baseline_alltime.merge(hexing_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.dropna(subset=['XEcefMeters_hexing']).reset_index(drop=True)
    if aignssname is not None:
        baseline_alltime = baseline_alltime.merge(aignss_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.merge(sppbl_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.merge(velbl_pd, on=merge_columns, suffixes=('', ''))
        baseline_alltime = baseline_alltime.dropna(subset=['XEcefMeters_AI']).reset_index(drop=True)  # 剔除无定位数据的行
    # baseline_alltime = baseline_alltime.merge(speed_smoothing_pd, on = merge_columns, suffixes = ('', '_smoothing'))
    baseline = baseline_alltime.drop_duplicates(subset=['UTCtime']).reset_index(drop=True) # 剔除重复行，如果重复就只提取第一行
    baseline = baseline.dropna(subset=['XEcefMeters_spp']).reset_index(drop=True)  # 剔除无定位数据的行

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
    if aignssname is not None:
        speed_spp = baseline[['VXEcefMeters_wlsbl', 'VYEcefMeters_wlsbl', 'VZEcefMeters_wlsbl']].to_numpy()
    else:
        speed_spp = baseline[['VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']].to_numpy()
    # speed_spp_smoothing = baseline[['VXEcefMeters_wls_smoothing','VYEcefMeters_wls_smoothing','VZEcefMeters_wls_smoothing']].to_numpy()

    """
    Solve with Kalman filter
    """
    if aignssname is None:
        zs = baseline[['XEcefMeters_spp','YEcefMeters_spp','ZEcefMeters_spp']].to_numpy()
        us = baseline[['VXEcefMeters_wls', 'VYEcefMeters_wls', 'VZEcefMeters_wls']].to_numpy()
    else:
        zs = baseline[['XEcefMeters_sppbl', 'YEcefMeters_sppbl', 'ZEcefMeters_sppbl']].to_numpy()
        us = baseline[['VXEcefMeters_wlsbl', 'VYEcefMeters_wlsbl', 'VZEcefMeters_wlsbl']].to_numpy()
    utctime = baseline['UTCtime'].to_numpy()
    # if '20250620' in filefolder:
    sigma_v = 3.1
    # else:
    #     sigma_v = 0.6 # 0.6
    sigma_x = 12.0 # 20
    initial_P = 0.3
    sigma_mahalanobis = 1000
    interrupt_time = 3 # s# 29.11 # Mahalanobis distance for rejecting innovation
    n, dim_x = zs.shape
    F = np.eye(3)  # Transition matrix
    H = np.eye(3)  # Measurement function
    R = sigma_x ** 1.250 * np.eye(3)  # Measurement noise
    # Initial state and covariance
    # x = zs[0, :3].T  # State
    P = initial_P ** 2 * np.eye(3)  # State covariance
    I = np.eye(dim_x)
    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    # count
    interrupt_num, outlier_vel_num, outlier_pos_num = 0,0,0
    # Kalman filtering
    for i, (u, z, utc) in enumerate(zip(us, zs, utctime)):
        Q = sigma_v ** 1.250 * np.eye(3)  # Process noise
        if i == 0:
            x = z.T
            x_kf[i] = x.T
            P_kf[i] = P
            utc_pre = utc
            u_pre = u
            continue
        elif (utc-utc_pre)>interrupt_time:
            print(f'interrupt in {utc}, time {utc - utc_pre}')
            x = z.T
            x_kf[i] = x.T
            P_kf[i] = P
            utc_pre = utc
            u_pre = u
            interrupt_num += 1
            continue

        # 当前速度缺失，不进行KF，用原始的spp位置
        if np.isnan(u).any():
            x = z.T
            # If prediction is not available, increase covariance
            P = P + 10 ** 2 * R
            utc_pre = utc
        else:
            # Prediction step
            if np.sqrt(np.sum(u**2)) > outlier_velocity: # 判断异常速度
                u = u_pre
                Q = Q * 10
                outlier_vel_num += 1
            x = F @ x + u.T * (utc-utc_pre)
            # Q = cov_v[i]
            P = (F @ P) @ F.T + Q
            # Check outliers for observation
            d = dist_err_XYZ(z,H @ x) # distance.mahalanobis(z, H @ x, np.linalg.inv(P))
            u_pre = u
            # Update step
            if d < sigma_mahalanobis:
                y = z.T - H @ x
                S = (H @ P) @ H.T + R
                K = (P @ H.T) @ np.linalg.inv(S)
                x = x + K @ y
                P = (I - (K @ H)) @ P
            else:
                # If observation update is not available, increase covariance
                P += 10 ** 2 * Q
                outlier_pos_num += 1
                print(f'Outlier pos in {utc}, diserr: {d}')

        x_kf[i] = x.T
        P_kf[i] = P
        utc_pre = utc
    baseline[['XEcefMeters_kf','YEcefMeters_kf','ZEcefMeters_kf']] = x_kf
    print(f'Interrupt num:{interrupt_num},Outlier vel num:{outlier_velocity},Outlier pos num:{outlier_pos_num}')

    """
    read rtk data from Haige
    """
    # rtk_path = f'{file_path}/{kinematicname}'
    # rtk_skiprows = 2
    # print(rtk_path)
    # rtk_data = []
    # max_len = 15
    # with open(rtk_path, 'r', encoding='utf-8') as file:
    #     for line in file: #读取文件（rtk_path）的每一行
    #         if 'GGA' in line: # 根据条件，如果该行有关键词，就提取出来，放到列表中
    #             split_line = [item.strip() for item in line.strip().split(',')]
    #             # 更新最长行的长度
    #             if len(split_line) == max_len:
    #                 rtk_data.append(split_line)
    #
    # raw_rtk_pd = pd.DataFrame(rtk_data)
    # rtk_pd = raw_rtk_pd.iloc[:, [1,2,4,6,7,8,9]]
    # rtk_pd.columns = ['UTCtime(hhmmss)','latitude(deg)_rtk', 'longitude(deg)_rtk','RTKstate','Satnum','HDOP','height(m)_rtk']
    # rtk_pd = rtk_pd[rtk_pd['latitude(deg)_rtk'] != '']
    # rtk_pd = rtk_pd.drop_duplicates(subset=['UTCtime(hhmmss)']).astype(float).reset_index(drop=True)
    # rtk_pd[['latitude(deg)_rtk', 'longitude(deg)_rtk']] = rtk_pd[['latitude(deg)_rtk', 'longitude(deg)_rtk']].apply(
    #     lambda x: x // 100 + (x % 100) / 60)
    # rtk_pd['UTCtime'] = rtk_pd['UTCtime(hhmmss)'].apply(lambda x: x//10000*3600+x%10000//100*60+x%100)

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
    if aignssname is not None:
        guess_xyz_aignss = np.array([baseline['XEcefMeters_AI'], baseline['YEcefMeters_AI'], baseline['ZEcefMeters_AI']])
        guess_aignss_XYZ2llh = np.array(
            [coord.ecef2geodetic(guess_xyz_aignss[:, i]) for i in range(guess_xyz_aignss.shape[1])])
        guess_xyz_sppbl = np.array([baseline['XEcefMeters_sppbl'], baseline['YEcefMeters_sppbl'], baseline['ZEcefMeters_sppbl']])
        guess_sppbl_XYZ2llh = np.array(
            [coord.ecef2geodetic(guess_xyz_sppbl[:, i]) for i in range(guess_xyz_sppbl.shape[1])])
    if ubloxname is not None and 'ublox' in methodlist:
        guess_xyz_ubx = np.array(
            [baseline['XEcefMeters_ublox'], baseline['YEcefMeters_ublox'], baseline['ZEcefMeters_ublox']])
        guess_ubx_XYZ2llh = np.array(
            [coord.ecef2geodetic(guess_xyz_ubx[:, i]) for i in range(guess_xyz_ubx.shape[1])])

    # 计算ENU坐标系的结果
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
    dist_ll_spp = vincenty_distance(guess_spp_XYZ2llh, true_llh.T)
    dist_ll_kf = vincenty_distance(guess_kf_XYZ2llh, true_llh.T)
    dist_h_spp = np.abs(guess_spp_XYZ2llh.T[2, :] - true_llh[2, :])
    dist_h_kf = np.abs(guess_kf_XYZ2llh.T[2, :] - true_llh[2, :])
    dist_ned_2D_spp, dist_ned_h_spp = dist_err_NED(truth_NED_gt,guess_NED_spp)
    dist_ned_2D_kf, dist_ned_h_kf = dist_err_NED(truth_NED_gt,guess_NED_kf)
    speed_err_xyz = dist_err_XYZ(speed_spp.T, speed_gt.T)
    if aignssname is not None:
        dist_err_xyz_aignss = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_aignss)
        dist_err_xyz_sppbl = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_sppbl)
        dist_ll_aignss = vincenty_distance(guess_aignss_XYZ2llh, true_llh.T)
        dist_ll_sppbl = vincenty_distance(guess_sppbl_XYZ2llh, true_llh.T)
        dist_h_aignss = np.abs(guess_aignss_XYZ2llh.T[2, :] - true_llh[2, :])
        dist_h_sppbl = np.abs(guess_sppbl_XYZ2llh.T[2, :] - true_llh[2, :])
    if 'ublox' in methodlist:
        dist_err_xyz_ubx = dist_err_XYZ(true_llh2XYZ.T, guess_xyz_ubx)
        dist_ll_ubx = vincenty_distance(guess_ubx_XYZ2llh, true_llh.T)
        dist_h_ubx = np.abs(guess_ubx_XYZ2llh.T[2, :] - true_llh[2, :])

    # speed_err_xyz_smoothing = dist_err_XYZ(speed_spp_smoothing.T, speed_gt.T)
    # 剔除异常值 现在数据有bug，先提取一部分看看
    if aignssname is not None:
        err_idx = np.where(dist_err_xyz_aignss > 300)
        keep_idx = dist_err_xyz_aignss <= 300
        dist_err_xyz_spp = dist_err_xyz_spp[keep_idx]
        dist_err_xyz_kf = dist_err_xyz_kf[keep_idx]
        dist_ned_2D_spp = dist_ned_2D_spp[keep_idx]
        dist_ned_2D_kf = dist_ned_2D_kf[keep_idx]
        dist_ned_h_spp = dist_ned_h_spp[keep_idx]
        dist_ned_h_kf = dist_ned_h_kf[keep_idx]
        dist_h_spp = dist_h_spp[keep_idx]
        dist_h_kf = dist_h_kf[keep_idx]
        dist_err_xyz_aignss = dist_err_xyz_aignss[keep_idx]
        dist_err_xyz_sppbl = dist_err_xyz_sppbl[keep_idx]
        dist_ll_aignss = dist_ll_aignss[keep_idx]
        dist_ll_sppbl = dist_ll_sppbl[keep_idx]
        dist_h_aignss = dist_h_aignss[keep_idx]
        dist_h_sppbl = dist_h_sppbl[keep_idx]
        if ubloxname is not None and 'ublox' in methodlist:
            dist_err_xyz_ubx = dist_err_xyz_ubx[keep_idx]
            dist_ll_ubx = dist_ll_ubx[keep_idx]
            dist_h_ubx = dist_h_ubx[keep_idx]
        baseline = baseline.drop(baseline.index[err_idx]) # 数据中剔除异常的列
    print(f'Outlier error num: {len(err_idx)}')

    # range
    # start_idx, end_idx = 3000 ,4000
    # dist_err_xyz_spp = dist_err_xyz_spp[start_idx: end_idx]
    # dist_err_xyz_kf = dist_err_xyz_kf[start_idx: end_idx]
    # dist_err_xyz_sppbl = dist_err_xyz_sppbl[start_idx: end_idx]
    # dist_err_xyz_aignss =dist_err_xyz_aignss[start_idx: end_idx]
    # dist_err_xyz_ubx = dist_err_xyz_ubx[start_idx: end_idx]
    # speed_err_xyz = speed_err_xyz[start_idx: end_idx]

    if record_dis:
        plt.figure()
        dis_err = dist_err_xyz_spp
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='SPP, average error={:.3}m'.format(np.nanmean(dist_err_xyz_spp)))
        dis_err = dist_err_xyz_kf
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='KF, average error={:.3}m'.format(np.nanmean(dist_err_xyz_kf)))
        if aignssname is not None:
            dis_err = dist_err_xyz_sppbl
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_err_xyz_sppbl)))
            dis_err = dist_err_xyz_aignss
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_err_xyz_aignss)))
        if 'ublox' in methodlist:
            dis_err = dist_err_xyz_ubx
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='ublox, average error={:.3}m'.format(np.nanmean(dist_err_xyz_ubx)))


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
        if 'ublox' in methodlist:
            plt.plot(dist_err_xyz_ubx, label='Ublox, average error={:.3}m'.format(np.nanmean(dist_err_xyz_ubx)))
        if aignssname is not None:
            plt.plot(dist_err_xyz_sppbl, label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_err_xyz_sppbl)))
            plt.plot(dist_err_xyz_aignss,  label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_err_xyz_aignss)))

        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Distance error of ecef in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Distance error (m)', fontsize=labelsize)
        #plt.ylim([0, 100])
        plt.savefig(f'{plt_folder}/Traj_100_ecef_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        dis_err = dist_ned_2D_spp
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='SPP, average error={:.3}m'.format(np.nanmean(dist_ned_2D_spp)))
        dis_err = dist_ned_2D_kf
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='KF, average error={:.3}m'.format(np.nanmean(dist_ned_2D_kf)))
        if aignssname is not None:
            dis_err = dist_ll_sppbl
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_ll_sppbl)))
            dis_err = dist_ll_aignss
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_ll_aignss)))
        if 'ublox' in methodlist and 'ublox' in methodlist:
            dis_err = dist_ll_ubx
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='ublox, average error={:.3}m'.format(np.nanmean(dist_ll_ubx)))

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
        if 'ublox' in methodlist:
            plt.plot(dist_ll_ubx, label='ublox, average error={:.3}m'.format(np.nanmean(dist_ll_ubx)))
        if aignssname is not None:
            plt.plot(dist_ll_sppbl, label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_ll_sppbl)))
            plt.plot(dist_ll_aignss, label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_ll_aignss)))

        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Horizontal error in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Horizontal error (m)', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Traj_org_ll_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        dis_err = dist_h_spp
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='SPP, average error={:.3}m'.format(np.nanmean(dist_h_spp)))
        dis_err = dist_h_kf
        dis_err = np.where(dis_err > err_max, err_max, dis_err)
        binnum = int(np.ceil(max(dis_err)))
        plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='KF, average error={:.3}m'.format(np.nanmean(dist_h_kf)))
        if aignssname is not None:
            dis_err = dist_h_sppbl
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_h_sppbl)))
            dis_err = dist_h_aignss
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_h_aignss)))
        if 'ublox' in methodlist and 'ublox' in methodlist:
            dis_err = dist_h_ubx
            dis_err = np.where(dis_err > err_max, err_max, dis_err)
            binnum = int(np.ceil(max(dis_err)))
            plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                     label='ublox, average error={:.3}m'.format(np.nanmean(dist_h_ubx)))

        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Elevation error distribution in {tripID}', fontsize=labelsize)
        plt.xlabel('Elevation error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Haige_h_disterrdistr_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(dist_h_spp, label='SPP, average error={:.3}m'.format(np.nanmean(dist_h_spp)))
        plt.plot(dist_h_kf, label='KF, average error={:.3}m'.format(np.nanmean(dist_h_kf)))
        if 'ublox' in methodlist:
            plt.plot(dist_h_ubx, label='ublox, average error={:.3}m'.format(np.nanmean(dist_h_ubx)))
        if aignssname is not None:
            plt.plot(dist_h_sppbl, label='SPP-bl, average error={:.3}m'.format(np.nanmean(dist_h_sppbl)))
            plt.plot(dist_h_aignss,  label='AI-GNSS, average error={:.3}m'.format(np.nanmean(dist_h_aignss)))
        plt.grid()
        plt.legend(fontsize=labelsize)
        plt.title(f'Elevation error in {tripID}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Elevation error (m)', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/Traj_org_h_{tripID}.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(speed_err_xyz, label=f'Velocity: error={np.nanmean(speed_err_xyz):.2}+{np.nanstd(speed_err_xyz):.2}m/s')
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

    baseline = baseline.reset_index(drop=True)
    data_truth_dic[tripID] = baseline
    tripIDlist.append(tripID)

    if len(pdop_data) > 0:
        pdop_dic[tripID] = dop_pd

# if save_file:
#     with open(base_directory + f'/env/processed_features_BDS_Haige_new.pkl', 'wb') as value_file:
#         pickle.dump(losfeature, value_file, True)
#     value_file.close()
#     with open(base_directory + f'/env/raw_baseline_Haige_new.pkl', 'wb') as value_file:
#         pickle.dump(data_truth_dic, value_file, True)
#     value_file.close()
#     with open(base_directory + f'/env/raw_gnss_multic_Haige_new.pkl', 'wb') as value_file:
#         pickle.dump(gnss_dic, value_file, True)
#     value_file.close()
#     # sat_summary_multic_all.to_csv(base_directory + f'/env/raw_satnum_BDS_Haige_new.csv', index=True)
#
#     tripID_df=pd.DataFrame(tripIDlist, columns=['tripID'])
#
#     tripID_df.to_csv(base_directory + f'/env/raw_tripID_BDS_Haige_new.csv', index=True)

if env_split:
    trajtypelist = ['openroad', 'canyon', 'overpass', 'highway', 'forest']
    method_name_list = {'spp':'SPP','sppbl':'SPP_bl','kf':'KF_bl','ublox':'ublox','hexing':'hexing','AI':'AI-AKF'}
    interrupt_time_thread = 100  # 中断统计时间，中断大于该时间截断为新轨迹
    continue_time = 1  # 连续时间小于该参数的轨迹剔除(只针对连续情况)
    ########## 统计数据
    disll_type = {}
    dish_type = {}
    disxyz_type = {}
    datatype = 'HaigeData'
    for method in methodlist:
        disxyz_type[method] = {}
        disll_type[method] = {}
        dish_type[method] = {}
        for type in trajtypelist:
            disll_type[method][type] = []
            dish_type[method][type] = []
            disxyz_type[method][type] = []

    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)

    for tripID, baseline in data_truth_dic.items():
        # tripID = '20250703-PM_1'
        # baseline = data_truth_dic[tripID]
        for datetime in trajtype_range_dic.keys():
            if datetime in tripID:
                range_dic = trajtype_range_dic[datetime]
        # 统计中断
        interrupt_time_all = 0
        interrupt_distance_all = 0
        reftimes_org = np.array(baseline['UTCtime'])  # utc时间
        for type, typerange in range_dic.items():
            print(f'Processing {type} in {tripID}')
            idx = np.where((typerange[0] <= reftimes_org) & (typerange[1] >= reftimes_org))  # 获取范围内索引
            try:
                trajidx = range(int(idx[0][0]), int(idx[0][-1]))
            except:
                continue
            ################## 统计中断
            count = 0
            interrupt_UTCtimes_list = []
            for index in trajidx:
                if index == int(idx[0][0]):
                    timestep_start = baseline['UTCtime'][index]  # start of the steptime
                    prepos = np.array([baseline['latitude(deg)'][index], baseline['longitude(deg)'][index],
                                       baseline['height(m)'][index]])
                    continue

                curpos = np.array([baseline['latitude(deg)'][index], baseline['longitude(deg)'][index],
                                   baseline['height(m)'][index]])
                dist_ll, _ = pmv.vdist(prepos[0], prepos[1], curpos[0], curpos[1])
                timestep_pre = baseline['UTCtime'][index - 1]
                timestep_cur = baseline['UTCtime'][index]
                interrupt_time = timestep_cur - timestep_pre
                if interrupt_time > interrupt_time_thread:
                    count += 1
                    if (timestep_pre - timestep_start) > continue_time:
                        interrupt_UTCtimes_list.append([timestep_start, timestep_pre])
                    timestep_start = timestep_cur
                    interrupt_time_all += interrupt_time
                    interrupt_distance_all += dist_ll
                    print(f'Count={count}, interrupt_dis={dist_ll:.2},'
                          f'interrupt_times={timestep_cur}-{timestep_pre}={interrupt_time}s')
                # 最后的idx，还要加上最后一段
                if index == int(idx[0][-1]) - 1:
                    interrupt_UTCtimes_list.append([timestep_start, timestep_cur + 1])
                prepos = curpos

            # Interrupt sequence trajectory dictionary
            for count, interrupt_UTCtime in enumerate(interrupt_UTCtimes_list):
                idx = np.where(
                    (interrupt_UTCtime[0] <= reftimes_org) & (interrupt_UTCtime[1] >= reftimes_org))
                if idx[0][-1] - idx[0][0] > 1:
                    balineline_split = baseline.iloc[idx]
                    if len(balineline_split) < continue_time:
                        continue

                    #### 统计误差
                    for method in methodlist:
                        true_XYZ = np.array(
                            [balineline_split['ecefX_gt'], balineline_split['ecefY_gt'], balineline_split['ecefZ_gt']])
                        true_llh = np.array(
                            [balineline_split['latitude(deg)'], balineline_split['longitude(deg)'],
                             balineline_split['height(m)']])
                        guess_xyz = np.array(
                            [balineline_split[f'XEcefMeters_{method}'], balineline_split[f'YEcefMeters_{method}'],
                             balineline_split[f'ZEcefMeters_{method}']])
                        guess_XYZ2llh = np.array([coord.ecef2geodetic(guess_xyz[:, i]) for i in range(guess_xyz.shape[1])])

                        dist_err_xyz = dist_err_XYZ(true_XYZ, guess_xyz)
                        dist_ll = vincenty_distance(guess_XYZ2llh, true_llh.T)
                        dist_h = np.abs(guess_XYZ2llh.T[2, :] - true_llh[2, :])
                        disll_type[method][f'{type[0:-1]}'].append(dist_ll)
                        disxyz_type[method][f'{type[0:-1]}'].append(dist_err_xyz)
                        dish_type[method][f'{type[0:-1]}'].append(dist_h)


    if plot_split_dis:
        err_max = 100
        no_trajtype = False # 判断该轨迹是否有对应环境
        for trajtype in trajtypelist:
            print(f'{trajtype}')
            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = disxyz_type[method][trajtype]
                if not value:
                    no_trajtype = True
                    break
                dis_err_or = np.concatenate(value)
                dis_err = np.where(dis_err_or > err_max, err_max, dis_err_or)
                binnum = int(np.ceil(max(dis_err)))
                plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                         label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                # plt.xlim([-0.5, 40])

            if no_trajtype:
                no_trajtype = False
                continue

            plt.grid()
            plt.legend(fontsize=labelsize - 2)
            plt.title(f'3D distance error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Distance error (m)', fontsize=labelsize)
            plt.ylabel('Proportion', fontsize=labelsize)
            # plt.savefig(f'{plt_folder}/3Derr_distribution_{type}.png', bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = disxyz_type[method][trajtype]
                dis_err_or = np.concatenate(value)
                plt.plot(dis_err_or, label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                result = statistic_data(dis_err_or)
                print(
                    f'3D error of {method}: average={result[0]:.2f}+{result[1]:.2f},67%={result[2]:.2f},95%={result[3]:.2f},max={result[4]:.2f},min={result[5]:.2f},CEP50={result[6]}:2f')
            plt.grid()
            plt.legend(fontsize=labelsize)
            plt.title(f'3D distance error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Step', fontsize=labelsize)
            plt.ylabel('Distance error (m)', fontsize=labelsize)
            if np.max(dis_err_or) > 200:
                plt.ylim([0, 200])
            plt.show()

        no_trajtype = False
        for trajtype in trajtypelist:
            print(f'{trajtype}')
            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = disll_type[method][trajtype]
                if not value:
                    no_trajtype = True
                    break
                dis_err_or = np.concatenate(value)
                dis_err = np.where(dis_err_or > err_max, err_max, dis_err_or)
                binnum = int(np.ceil(max(dis_err)))
                plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                         label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                # plt.xlim([-0.5, 40])

            if no_trajtype:
                no_trajtype = False
                continue

            plt.grid()
            plt.legend(fontsize=labelsize - 2)
            plt.title(f'Horizontal error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Horizontal error (m)', fontsize=labelsize)
            plt.ylabel('Proportion', fontsize=labelsize)
            # plt.savefig(f'{plt_folder}/3Derr_distribution_{type}.png', bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = disll_type[method][trajtype]
                dis_err_or = np.concatenate(value)
                plt.plot(dis_err_or, label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                result = statistic_data(dis_err_or)
                print(f'2D error of {method}: average={result[0]:.2f}+{result[1]:.2f},67%={result[2]:.2f},95%={result[3]:.2f},max={result[4]:.2f},min={result[5]:.2f},CEP50={result[6]}:2f')
            plt.grid()
            plt.legend(fontsize=labelsize)
            plt.title(f'Horizontal error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Step', fontsize=labelsize)
            plt.ylabel('Horizontal error (m)', fontsize=labelsize)
            if np.max(dis_err_or) > 200:
                plt.ylim([0, 200])
            plt.show()

        no_trajtype = False
        for trajtype in trajtypelist:
            print(f'{trajtype}')
            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = dish_type[method][trajtype]
                if not value:
                    no_trajtype = True
                    break
                dis_err_or = np.concatenate(value)
                dis_err = np.where(dis_err_or > err_max, err_max, dis_err_or)
                binnum = int(np.ceil(max(dis_err)))
                plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                         label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                # plt.xlim([-0.5, 40])

            if no_trajtype:
                no_trajtype = False
                continue

            plt.grid()
            plt.legend(fontsize=labelsize - 2)
            plt.title(f'Elevation error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Elevation error (m)', fontsize=labelsize)
            plt.ylabel('Proportion', fontsize=labelsize)
            # plt.savefig(f'{plt_folder}/3Derr_distribution_{type}.png', bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(6, 5))
            for method in methodlist:
                value = dish_type[method][trajtype]
                dis_err_or = np.concatenate(value)
                plt.plot(dis_err_or, label=f'{method_name_list[method]}, mean={np.mean(dis_err_or):.3}(std:{np.std(dis_err_or):.3})m')
                result = statistic_data(dis_err_or)
                print(
                    f'Elevation error of {method}: average={result[0]:.2f}+{result[1]:.2f},67%={result[2]:.2f},95%={result[3]:.2f},max={result[4]:.2f},min={result[5]:.2f},CEP50={result[6]}:2f')
            print(f'samples = {len(dis_err_or)}')
            plt.grid()
            plt.legend(fontsize=labelsize)
            plt.title(f'Elevation error in {trajtype}', fontsize=labelsize)
            plt.xlabel('Step', fontsize=labelsize)
            plt.ylabel('Elevation error (m)', fontsize=labelsize)
            if np.max(dis_err_or) > 200:
                plt.ylim([0, 200])
            plt.show()


print('Processing finish !')


