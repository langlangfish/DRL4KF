# import library
# import math
# import numpy as np
# from math import radians, cos, sin, asin, sqrt
# from haversine import haversine
# import pandas as pd
# import sys
# sys.path.append("../../..")
# import gnss_lib.coordinates as coord
# import pymap3d.vincenty as pmv
# from datetime import datetime, timedelta
# import scipy.optimize
# from tqdm.auto import tqdm
# import matplotlib.pyplot as plt
# import pymap3d as pm
# from scipy.interpolate import InterpolatedUnivariateSpline
# from scipy.spatial import distance
from src.utils.package import np, sin, cos, math, radians, asin, sqrt, datetime, timedelta, pmv, pd, tqdm, scipy, plt, \
    haversine, pm
from src.utils.gnss_lib import coordinates as coord

CLIGHT = 299_792_458
B1I_carrier = 1561.098 * 1e6
B1C_carrier = 1575.42 * 1e6
RE_WGS84 = 6_378_137  # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)
v_up_th = 2.6  # m/s  2.0 -> 2.6
v_out_sigma = 3.0  # m/s
position_Hz = 1  # hz
outlier_velocity = 37  # m/s（国内最高限速33m/s）


def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0 - expFactor)
            J += (expFactor)
            rate = expFactor / J
            expRaw = (1 - rate) * expRaw
            expRaw += rate * data[i][j]
            expRawRewards[i, j] = expRaw
    return expRawRewards


def exp_average_list(data, expFactor=0.1):
    expRawRewards = np.zeros(len(data))
    expRaw = 0.0
    J = 0.0
    for j in range(len(data)):
        J *= (1.0 - expFactor)
        J += (expFactor)
        rate = expFactor / J
        expRaw = (1 - rate) * expRaw
        expRaw += rate * data[j]
        expRawRewards[j] = expRaw
    return expRawRewards


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


def satellite_selection(df, column):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df[column].notnull()
    # idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['EA'] * 180 / math.pi > 10.0  # elevation angle (deg)
    idx &= df['CNR'] > 15.0  # C/N0 (dB-Hz)
    return df[idx]


def gps_week_seconds_to_datetime(gps_week, gps_seconds):
    gps_epoch = datetime(1980, 1, 6)  # GPS epoch
    elapsed = timedelta(weeks=int(gps_week), seconds=float(gps_seconds))
    return gps_epoch + elapsed


def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


def dist_err_XYZ(true_XYZ, guess_XYZ_bl):
    tmp = true_XYZ - guess_XYZ_bl
    dist_err = np.sqrt(np.sum(tmp ** 2, axis=0))
    return dist_err


def dist_err_LLH(true_llh, guess_llh_bl):
    true_ecef = coord.geodetic2ecef(true_llh.T)
    guess_ecef = coord.geodetic2ecef(guess_llh_bl.T)
    dist_err = dist_err_XYZ(true_ecef.T, guess_ecef.T)
    return dist_err


def dist_err_NED(truth_NED_gt, guess_NED):
    tmp = truth_NED_gt.T[:2] - guess_NED.T[:2]
    dist_ned_2D = np.sqrt(np.sum(tmp ** 2, axis=0))
    dist_ned_h = np.abs(truth_NED_gt.T[2] - guess_NED.T[2])
    return dist_ned_2D, dist_ned_h


def smooth_velocity(df, threshold):
    result_df = df.copy()
    cnt = 0
    for i in range(1, len(df)):
        if (result_df.iloc[i, :3].isna().any()) or (result_df.iloc[i - 1, :3].isna().any()):
            continue
        cur_vel = df.iloc[i, :3].values
        # if abs(diff_vel) > threshold:
        if np.sqrt(np.sum(cur_vel ** 2)) > threshold:
            # 将当前行替换为上一行的值
            result_df.iloc[i, :3] = result_df.iloc[i - 1, :3]
            cnt += 1
    print(f'Outlier velocity num: {cnt}')
    return result_df


def smooth_position(df, threshold):
    result_df = df.copy()
    cnt = 0
    for i in range(1, len(df)):
        if (result_df.iloc[i, :3].isna().any()) or (result_df.iloc[i - 1, :3].isna().any()):
            continue
        cur_pos = df.iloc[i, :3].values
        pre_pos = df.iloc[i - 1, :3].values
        dist_xyz = np.sqrt(np.sum((cur_pos - pre_pos) ** 2))
        # if abs(diff_vel) > threshold:
        if dist_xyz > threshold:
            # 将当前行替换为上一行的值
            result_df.iloc[i, :3] = result_df.iloc[i - 1, :3]
            cnt += 1
    print(f'Outlier position num: {cnt}')
    return result_df


def smooth_pos_vel(df_pos, df_vel, threshold):
    result_pos_df = df_pos.copy()
    result_vel_df = df_vel.copy()
    cnt_pos = 0
    cnt_vel = 0
    for i in range(1, len(result_pos_df)):
        if (result_pos_df.iloc[i, :3].isna().any()) or (result_pos_df.iloc[i - 1, :3].isna().any()):
            continue
        cur_pos = df_pos.iloc[i, :3].values
        pre_pos = df_pos.iloc[i - 1, :3].values
        cur_vel = df_vel.iloc[i, :3].values
        cur_UTC = df_pos.iloc[i, 3]
        pre_UTC = df_pos.iloc[i - 1, 3]
        dist_xyz = np.sqrt(np.sum((cur_pos - pre_pos) ** 2))
        # if abs(diff_vel) > threshold:
        if (dist_xyz > threshold) and (cur_UTC - pre_UTC == 1 / position_Hz):
            # 将当前行替换为上一行的位置用动力学推
            result_pos_df.iloc[i, :3] = result_pos_df.iloc[i - 1, :3] + cur_vel * (
                    cur_UTC - pre_UTC)  # 这地方有点问题，应该再加上一个速度比较合理
            cnt_pos += 1
        if np.sqrt(np.sum(cur_vel ** 2)) > threshold:
            # 将当前行替换为上一行的值
            result_vel_df.iloc[i, :3] = result_vel_df.iloc[i - 1, :3]
            cnt_vel += 1
    print(f'Outlier position num: {cnt_pos}')
    print(f'Outlier velocity num: {cnt_vel}')
    return result_pos_df, result_vel_df


def smooth_vel(df_vel, threshold):
    result_vel_df = df_vel.copy()
    cnt_pos = 0
    cnt_vel = 0
    for i in range(1, len(result_vel_df)):
        if (result_vel_df.iloc[i, :3].isna().any()) or (result_vel_df.iloc[i - 1, :3].isna().any()):
            continue
        cur_vel = df_vel.iloc[i, :3].values
        # if abs(diff_vel) > threshold:
        if np.sqrt(np.sum(cur_vel ** 2)) > threshold:
            # 将当前行替换为上一行的值
            result_vel_df.iloc[i, :3] = result_vel_df.iloc[i - 1, :3]
            cnt_vel += 1
    print(f'Outlier velocity num: {cnt_vel}')
    return result_vel_df


def list2pd(data, max_len, index_list, colums):
    data_padded = [line + [np.nan] * (max_len - len(line)) for line in data]
    raw_pd = pd.DataFrame(data_padded)
    process_pd = raw_pd.iloc[:, index_list].replace('', np.nan).apply(pd.to_numeric, errors='coerce')
    process_pd.columns = colums
    return process_pd


def diff_velocity(u_pre, u_cur):
    return np.sqrt(np.sum(u_pre ** 2)) - np.sqrt(np.sum(u_cur ** 2))


# Compute line-of-sight vector from user to satellite
def los_vector(xusr, xsat):
    """
    Args:
        xusr : user position in ECEF (m)
        xsat : satellite position in ECEF (m)
    Returns:
        u: unit line-of-sight vector in ECEF (m)
        rng: distance between user and satellite (m)
    """
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng

    return u, rng.reshape(-1)


"""
solve velocity with wls
"""


def WLS_velocity_solution(tripID, gnss_pd, spp_pd, speed_outlier=True):
    UTCtime = gnss_pd['UTCtime'].unique()
    nepoch = len(UTCtime)
    v0 = np.zeros(4)
    v_wls = np.full([nepoch, 3], np.nan)
    v_wls_or = np.full([nepoch, 3], np.nan)  # 未平滑的速度
    cov_v = np.full([nepoch, 3, 3], np.nan)
    speed_pd = pd.DataFrame(columns=['UTCtime', 'VXEcefMeters_self', 'VYEcefMeters_self', 'VZEcefMeters_self'])
    speed_or_pd = pd.DataFrame(columns=['UTCtime', 'VXEcefMeters_self', 'VYEcefMeters_self', 'VZEcefMeters_self'])
    err_cnt = 0
    for i, (t_utc, df) in enumerate(tqdm(gnss_pd.groupby('UTCtime'), total=nepoch)):
        df_prr = satellite_selection(df, 'dopple_m_s')
        doppler = df_prr['dopple_m_s'].to_numpy()
        prr = CLIGHT / B1I_carrier * doppler
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()
        # 高度角定权
        W_ea = 0.09 + (0.09 / np.sin(df_prr['EA'].to_numpy()) ** 2)
        Wv = np.diag(1 / W_ea)  # np.diag(df_prr['EA'].to_numpy())
        # 载噪比定权
        # W_cn = (CLIGHT/B1I_carrier / (2*math.pi))**2 * 10**(-df_prr['CNR']/10)
        # Wv = np.diag(W_cn)
        try:
            x0 = spp_pd[spp_pd['UTCtime'] == t_utc][
                ['XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']].to_numpy().reshape(3, )
            if np.any(np.isnan(x0)):
                continue
        except:
            continue
        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            try:
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            except:
                continue
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
                v_wls[i, :] = opt.x[:3]
                v_wls_or[i, :] = opt.x[:3]
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v_wls_or[i, :] = opt.x[:3]
                v0 = opt.x
            # speed smoothing
            if speed_outlier:
                x_llh = np.array(pm.ecef2geodetic(x0[0], x0[1], x0[2])).T
                v_enu = np.array(pm.ecef2enuv(v_wls[i, 0], v_wls[i, 1], v_wls[i, 2], x_llh[0], x_llh[1])).T
                # 两种剔除异常方式：全速度异常剔除，高程速度异常剔除
                # np.sqrt(np.sum(v_wls[i]**2)) > outlier_velocity
                # np.abs(v_enu[2]) > v_up_th
                if np.abs(v_enu[2]) > v_up_th:  #
                    v_wls[i, :] = (v_wls[i - 1, :] + v_wls[i - 2, :]) * 0.5
                    cov_v[i, :, :] = v_out_sigma ** 2 * np.eye(3)
                    err_cnt = err_cnt + 1
            speed_pd.loc[i] = [t_utc, v_wls[i, 0], v_wls[i, 1], v_wls[i, 2]]
            speed_or_pd.loc[i] = [t_utc, v_wls_or[i, 0], v_wls_or[i, 1], v_wls_or[i, 2]]
    print(f'{tripID}, epoch num={nepoch}, Outlier velocity num={err_cnt}, ratio={err_cnt / nepoch}')
    columns_to_interpolate = ['VXEcefMeters_self', 'VYEcefMeters_self', 'VZEcefMeters_self']
    speed_pd[columns_to_interpolate] = speed_pd[columns_to_interpolate].interpolate(method='linear')
    speed_or_pd[columns_to_interpolate] = speed_or_pd[columns_to_interpolate].interpolate(method='linear')
    return speed_pd, speed_or_pd, cov_v


def Kalmanfilter(zs, us, baseline, sigma_v, sigma_x, initial_P, outlier_velocity):
    utctime = baseline['UTCtime'].to_numpy()
    # if '20250620' in filefolder:

    sigma_mahalanobis = 1000  # 29.11 # Mahalanobis distance for rejecting innovation
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
    # Kalman filtering
    for i, (u, z, utc) in enumerate(zip(us, zs, utctime)):
        Q = sigma_v ** 1.250 * np.eye(3)  # Process noise
        if (i == 0) or (utc - utc_pre) > 3:
            x = z.T
            x_kf[i] = x.T
            P_kf[i] = P
            utc_pre = utc
            u_pre = u
            continue

        # 当前速度缺失，不进行KF，用原始的spp位置
        if np.isnan(u).any():
            x = z.T
            # If prediction is not available, increase covariance
            P = P + 10 ** 2 * R
            utc_pre = utc
        else:
            # Prediction step
            # 两种剔除异常方式：全速度异常剔除，高程速度异常剔除
            # np.sqrt(np.sum(u**2)) > outlier_velocity: # 判断异常速度
            # np.abs(v_enu[2]) > v_up_th
            x_llh = np.array(pm.ecef2geodetic(z[0], z[1], z[2])).T
            v_enu = np.array(pm.ecef2enuv(u[0], u[1], u[2], x_llh[0], x_llh[1])).T
            if np.sqrt(np.sum(u ** 2)) > outlier_velocity:  #
                u = u_pre
                Q = Q  # * 10
            x = F @ x + u.T * (utc - utc_pre)
            # Q = cov_v[i]
            P = (F @ P) @ F.T + Q
            # Check outliers for observation
            d = dist_err_XYZ(z, H @ x)
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

        x_kf[i] = x.T
        P_kf[i] = P
        utc_pre = utc
    return x_kf


"""
saga-husa-AKF
"""


def Saga_husa_AKF(zs, us, baseline, sigma_v, sigma_x, initial_P, outlier_velocity, lamb):
    utctime = baseline['UTCtime'].to_numpy()
    sigma_mahalanobis = 100000000000  # 29.11 # Mahalanobis distance for rejecting innovation
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
    Q = sigma_v ** 1.250 * np.eye(3)  # Process noise
    # Kalman filtering
    for i, (u, z, utc) in enumerate(zip(us, zs, utctime)):
        if (i == 0) or (utc - utc_pre) > 3:
            x = z.T
            x_kf[i] = x.T
            P_kf[i] = P
            utc_pre = utc
            u_pre = u
            continue

        # 当前速度缺失，不进行KF，用原始的spp位置
        if np.isnan(u).any():
            x = z.T
            # If prediction is not available, increase covariance
            P = P + 10 ** 2 * R
            utc_pre = utc
        else:
            # Prediction step
            if np.sqrt(np.sum(u ** 2)) > outlier_velocity:  # 判断异常速度
                u = u_pre
                Q = Q * 10
            P_1 = P.copy()
            x = F @ x + u.T * (utc - utc_pre)
            # adaptive Q
            P = (F @ P) @ F.T + Q
            u_pre = u
            # Update step
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
            # adaptive Q
            d_t = lamb * (1 / (i + 1))  # (1 - lamb) / (1 - lamb**(i+1))
            y = y.reshape(-1, 1)
            Q_temp = K @ y @ y.T @ K.T + P - P_1
            Q = (1 - d_t) * Q + d_t * Q_temp

        x_kf[i] = x.T
        P_kf[i] = P
        utc_pre = utc
    return x_kf


def plot_dist(dist_err_or, err_max, method):
    dis_err = dist_err_or
    dis_err = np.where(dis_err > err_max, err_max, dis_err)
    binnum = int(np.ceil(max(dis_err)))
    plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
             label=f'{method}, average error={np.nanmean(dist_err_or):.3}m')


# Compute pseudorange rate residuals
def prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange rate residuals
    """
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat - v[:3]) * u, axis=1) \
           + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                              - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W


# Compute Jacobian matrix
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return W @ J


def cal_distance(row):
    """
    计算两个经纬度点之间的距离
    """
    long1 = row['LongitudeDegrees_truth']
    lat1 = row['LatitudeDegrees_truth']
    long2 = row['lngDeg_RLpredict']
    lat2 = row['latDeg_RLpredict']
    long3 = row['LongitudeDegrees']
    lat3 = row['LatitudeDegrees']
    g1 = (lat1, long1)
    g2 = (lat2, long2)
    g3 = (lat3, long3)
    # g1 = (long1, lat1)
    # g2 = (long2, lat2)
    # g3 = (long3, lat3)
    ret1 = haversine(g1, g2, unit='m')
    ret2 = haversine(g1, g3, unit='m')
    result1 = "%.7f" % ret1
    result2 = "%.7f" % ret2
    return result1, result2


def cal_distance_ecef(test, baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    y1 = test['ecefY']
    x1 = test['ecefX']
    z1 = test['ecefZ']
    y2 = test['Y_RLpredict']
    x2 = test['X_RLpredict']
    z2 = test['Z_RLpredict']
    if baseline_mod == 'bl':
        y3 = test['YEcefMeters_bl']
        x3 = test['XEcefMeters_bl']
        z3 = test['ZEcefMeters_bl']
    elif baseline_mod == 'wls':
        y3 = test['YEcefMeters_wls']
        x3 = test['XEcefMeters_wls']
        z3 = test['ZEcefMeters_wls']
    elif baseline_mod == 'bds':
        y3 = test['YEcefMeters_bds']
        x3 = test['XEcefMeters_bds']
        z3 = test['ZEcefMeters_bds']
    elif baseline_mod == 'kf':
        y3 = test['YEcefMeters_kf']
        x3 = test['XEcefMeters_kf']
        z3 = test['ZEcefMeters_kf']
    elif baseline_mod == 'kf_igst':
        y3 = test['YEcefMeters_kf_igst']
        x3 = test['XEcefMeters_kf_igst']
        z3 = test['ZEcefMeters_kf_igst']
    elif baseline_mod == 'single':
        y3 = test['YEcefMeters_single']
        x3 = test['XEcefMeters_single']
        z3 = test['ZEcefMeters_single']
    elif baseline_mod == 'rtk':
        y3 = test['YEcefMeters_rtk']
        x3 = test['XEcefMeters_rtk']
        z3 = test['ZEcefMeters_rtk']
    llh1 = coord.ecef2geodetic([x1, y1, z1])
    llh3 = coord.ecef2geodetic([x3, y3, z3])
    llerr2 = haversine((llh1[0], llh1[1]), (llh3[0], llh3[1]), unit='m')
    # llerr2 = pmv.vdist(llh1[0],llh1[1], llh3[0],llh3[1])
    herr2 = (llh3[-1] - llh1[-1])
    herrabs2 = np.abs(llh3[-1] - llh1[-1])
    result1 = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    result2 = np.sqrt(((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2))
    if np.isnan(x2) or np.isnan(y2) or np.isnan(z2):
        xerr1 = np.nan
        yerr1 = np.nan
        zerr1 = np.nan
        llerr1 = np.nan
        herr1 = np.nan
        herrabs1 = np.nan
    else:
        xerr1 = np.sqrt(((x2 - x1) ** 2))
        yerr1 = np.sqrt(((y2 - y1) ** 2))
        zerr1 = np.sqrt(((z2 - z1) ** 2))
        llh2 = coord.ecef2geodetic([x2, y2, z2])
        llerr1 = haversine((llh1[0], llh1[1]), (llh2[0], llh2[1]), unit='m')
        # llerr1 = pmv.vdist(llh1[0],llh1[1],llh2[0],llh2[1])
        herr1 = (llh2[-1] - llh1[-1])
        herrabs1 = np.abs(llh2[-1] - llh1[-1])
    xerr2 = np.sqrt(((x3 - x1) ** 2))
    yerr2 = np.sqrt(((y3 - y1) ** 2))
    zerr2 = np.sqrt(((z3 - z1) ** 2))

    return result1, result2, xerr1, yerr1, zerr1, xerr2, yerr2, zerr2, llerr1, herr1, llerr2, herr2, herrabs1, herrabs2


def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a ** 0.5)
    return dist


def percentile50(x):
    return np.percentile(x, 50)


def percentile95(x):
    return np.percentile(x, 95)


def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg': 'latDeg_gt', 'lngDeg': 'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score


def recording_results(data_truth_dic, trajdata_range, tripIDlist, logdirname):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            test = pd_train.loc[:, ['LongitudeDegrees_truth', 'LatitudeDegrees_truth',
                                    'lngDeg_RLpredict', 'latDeg_RLpredict', 'LongitudeDegrees', 'LatitudeDegrees']]
            test['rl_distance'] = test.apply(lambda test: cal_distance(test)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance(test)[1], axis=1)
            test['error'] = test['rl_distance'].astype(
                float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            if train_tripIDnum > trajdata_range[0]:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * \
                                     test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * \
                                     test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * \
                                    test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * \
                                    test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg',
                    [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                     error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')


def recording_results_ecef(data_truth_dic, trajdata_range, tripIDlist, logdirname, baseline_mod, traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    pd_gen = False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            pd_train = pd_train[pd_train['lat_RLpredict'].notnull()]
            if traj_record:
                # record rl traj
                record_columns = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)',
                                  'latitude(deg)_single', 'longitude(deg)_single',
                                  'height(m)_single', 'LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters',
                                  'lat_RLpredict', 'lon_RLpredict', 'alt_RLpredict']
                pd_record = pd_train[record_columns]
                pd_record = pd_record[pd_record['lat_RLpredict'].notnull()]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum]}.csv', index=True)
            if baseline_mod == 'bl':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bl', 'YEcefMeters_bl', 'ZEcefMeters_bl']]
            elif baseline_mod == 'wls':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_wls', 'YEcefMeters_wls', 'ZEcefMeters_wls']]
            elif baseline_mod == 'bds':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bds', 'YEcefMeters_bds', 'ZEcefMeters_bds']]
            elif baseline_mod == 'kf':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
            elif baseline_mod == 'kf_igst':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']]
            elif baseline_mod == 'single':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_single', 'YEcefMeters_single', 'ZEcefMeters_single']]
            elif baseline_mod == 'rtk':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_rtk', 'YEcefMeters_rtk', 'ZEcefMeters_rtk']]

            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(
                float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)
            rl_lldistance = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[8], axis=1)
            test['rl_lldistance'] = rl_lldistance
            test['rl_hdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[9], axis=1)
            test['count_rl_lldistance'] = test['rl_lldistance'].astype(float)
            test['count_rl_hdistance'] = test['rl_hdistance'].astype(float)
            or_lldistance = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[10], axis=1)
            test['or_lldistance'] = or_lldistance
            test['or_hdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[11], axis=1)
            test['count_or_lldistance'] = test['or_lldistance'].astype(float)
            test['count_or_hdistance'] = test['or_hdistance'].astype(float)
            test['rl_habsdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[12], axis=1)
            test['count_rl_habsdistance'] = test['rl_habsdistance'].astype(float)
            test['or_habsdistance'] = test.apply(lambda test: cal_distance_ecef(test, baseline_mod)[13], axis=1)
            test['count_or_habsdistance'] = test['or_habsdistance'].astype(float)

            print(
                f'RL LL distance: {np.mean(rl_lldistance):4.3f} + {np.std(rl_lldistance):4.3f}, OR LL distances: {np.mean(or_lldistance):4.3f} + {np.std(or_lldistance):4.3f}.')
            rl_xdistance_mean = np.mean(test['rl_xdistance'])
            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID': tripIDlist[train_tripIDnum],
                           'rl_xdistance_mean': np.mean(test['rl_xdistance']),
                           'rl_ydistance_mean': np.mean(test['rl_ydistance']),
                           'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                           'rl_xdistance_std': np.std(test['rl_xdistance']),
                           'rl_ydistance_std': np.std(test['rl_ydistance']),
                           'rl_zdistance_std': np.std(test['rl_zdistance']),
                           'rl_xdistance_min': np.nanmin(test['rl_xdistance']),
                           'rl_ydistance_min': np.nanmin(test['rl_ydistance']),
                           'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                           'rl_xdistance_max': np.nanmax(test['rl_xdistance']),
                           'rl_ydistance_max': np.nanmax(test['rl_ydistance']),
                           'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                           'or_xdistance_mean': np.mean(test['or_xdistance']),
                           'or_ydistance_mean': np.mean(test['or_ydistance']),
                           'or_zdistance_mean': np.mean(test['or_zdistance']),
                           'or_xdistance_std': np.std(test['or_xdistance']),
                           'or_ydistance_std': np.std(test['or_ydistance']),
                           'or_zdistance_std': np.std(test['or_zdistance']),
                           'or_xdistance_min': np.nanmin(test['or_xdistance']),
                           'or_ydistance_min': np.nanmin(test['or_ydistance']),
                           'or_zdistance_min': np.nanmin(test['or_zdistance']),
                           'or_xdistance_max': np.nanmax(test['or_xdistance']),
                           'or_ydistance_max': np.nanmax(test['or_ydistance']),
                           'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),
                           'rl_llerr_std': np.std(test['rl_lldistance']),
                           'rl_llerr_min': np.nanmin(test['rl_lldistance']),
                           'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                           'rl_herr_mean': np.mean(test['rl_hdistance']), 'rl_herr_std': np.std(test['rl_hdistance']),
                           'rl_herr_min': np.nanmin(test['rl_hdistance']),
                           'rl_herr_max': np.nanmax(test['rl_hdistance']),
                           'rl_habserr_mean': np.mean(test['rl_habsdistance']),
                           'rl_habserr_std': np.std(test['rl_habsdistance']),
                           'rl_habserr_min': np.nanmin(test['rl_habsdistance']),
                           'rl_habserr_max': np.nanmax(test['rl_habsdistance']),
                           'or_llerr_mean': np.mean(test['or_lldistance']),
                           'or_llerr_std': np.std(test['or_lldistance']),
                           'or_llerr_min': np.nanmin(test['or_lldistance']),
                           'or_llerr_max': np.nanmax(test['or_lldistance']),
                           'or_herr_mean': np.mean(test['or_hdistance']), 'or_herr_std': np.std(test['or_hdistance']),
                           'or_herr_min': np.nanmin(test['or_hdistance']),
                           'or_herr_max': np.nanmax(test['or_hdistance']),
                           'or_habserr_mean': np.mean(test['or_habsdistance']),
                           'or_habserr_std': np.std(test['or_habsdistance']),
                           'or_habserr_min': np.nanmin(test['or_habsdistance']),
                           'or_habserr_max': np.nanmax(test['or_habsdistance']),
                           }

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID': tripIDlist[train_tripIDnum],
                           'rl_xdistance_mean': np.mean(test['rl_xdistance']),
                           'rl_ydistance_mean': np.mean(test['rl_ydistance']),
                           'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                           'rl_xdistance_std': np.std(test['rl_xdistance']),
                           'rl_ydistance_std': np.std(test['rl_ydistance']),
                           'rl_zdistance_std': np.std(test['rl_zdistance']),
                           'rl_xdistance_min': np.nanmin(test['rl_xdistance']),
                           'rl_ydistance_min': np.nanmin(test['rl_ydistance']),
                           'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                           'rl_xdistance_max': np.nanmax(test['rl_xdistance']),
                           'rl_ydistance_max': np.nanmax(test['rl_ydistance']),
                           'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                           'or_xdistance_mean': np.mean(test['or_xdistance']),
                           'or_ydistance_mean': np.mean(test['or_ydistance']),
                           'or_zdistance_mean': np.mean(test['or_zdistance']),
                           'or_xdistance_std': np.std(test['or_xdistance']),
                           'or_ydistance_std': np.std(test['or_ydistance']),
                           'or_zdistance_std': np.std(test['or_zdistance']),
                           'or_xdistance_min': np.nanmin(test['or_xdistance']),
                           'or_ydistance_min': np.nanmin(test['or_ydistance']),
                           'or_zdistance_min': np.nanmin(test['or_zdistance']),
                           'or_xdistance_max': np.nanmax(test['or_xdistance']),
                           'or_ydistance_max': np.nanmax(test['or_ydistance']),
                           'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),
                           'rl_llerr_std': np.std(test['rl_lldistance']),
                           'rl_llerr_min': np.nanmin(test['rl_lldistance']),
                           'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                           'rl_herr_mean': np.mean(test['rl_hdistance']), 'rl_herr_std': np.std(test['rl_hdistance']),
                           'rl_herr_min': np.nanmin(test['rl_hdistance']),
                           'rl_herr_max': np.nanmax(test['rl_hdistance']),
                           'rl_habserr_mean': np.mean(test['rl_habsdistance']),
                           'rl_habserr_std': np.std(test['rl_habsdistance']),
                           'rl_habserr_min': np.nanmin(test['rl_habsdistance']),
                           'rl_habserr_max': np.nanmax(test['rl_habsdistance']),
                           'or_llerr_mean': np.mean(test['or_lldistance']),
                           'or_llerr_std': np.std(test['or_lldistance']),
                           'or_llerr_min': np.nanmin(test['or_lldistance']),
                           'or_llerr_max': np.nanmax(test['or_lldistance']),
                           'or_herr_mean': np.mean(test['or_hdistance']), 'or_herr_std': np.std(test['or_hdistance']),
                           'or_herr_min': np.nanmin(test['or_hdistance']),
                           'or_herr_max': np.nanmax(test['or_hdistance']),
                           'or_habserr_mean': np.mean(test['or_habsdistance']),
                           'or_habserr_std': np.std(test['or_habsdistance']),
                           'or_habserr_min': np.nanmin(test['or_habsdistance']),
                           'or_habserr_max': np.nanmax(test['or_habsdistance']),
                           }
                xyz_distance_pd = pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen = True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * \
                                     test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * \
                                     test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * \
                                    test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * \
                                    test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg',
                    [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                     error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_distances.csv', index=True)
    ll_err = xyz_distance_pd.loc['rl_llerr_mean'].values
    all_dis = np.mean(ll_err)

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m, rl_distance_avg: {rl_distances_mean_all / num_total_rl:0.3f}+{rl_distances_std_all / num_total_rl:0.3f},'
        f'or_distance_avg: {or_distances_mean_all / num_total_or:0.3f}+{or_distances_std_all / num_total_or:0.3f}.')

    return all_dis
