# import library
import numpy as np
from math import radians, cos, sin, asin, sqrt
from haversine import haversine
import pandas as pd
import gnss_lib.coordinates as coord

def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0-expFactor)
            J += (expFactor)
            rate = expFactor/J
            expRaw = (1-rate)*expRaw
            expRaw += rate*data[i][j]
            expRawRewards[i, j] = expRaw
    return expRawRewards

def exp_average_list(data, expFactor=0.1):
    expRawRewards = np.zeros(len(data))
    expRaw = 0.0
    J = 0.0
    for j in range(len(data)):
        J *= (1.0-expFactor)
        J += (expFactor)
        rate = expFactor/J
        expRaw = (1-rate)*expRaw
        expRaw += rate*data[j]
        expRawRewards[j] = expRaw
    return expRawRewards


def extract_between(string, start_char, end_char):
    start_index = string.find(start_char)
    end_index = string.find(end_char, start_index)

    if start_index != -1 and end_index != -1:
        # 提取 start_char 和 end_char 之间的字符串
        return string[start_index + 1:end_index]
    else:
        return None  # 如果没有找到字符，返回 None

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

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
    g1 = (lat1,long1)
    g2 = (lat2,long2)
    g3 = (lat3,long3)
    # g1 = (long1, lat1)
    # g2 = (long2, lat2)
    # g3 = (long3, lat3)
    ret1 = haversine(g1, g2, unit='m')
    ret2 = haversine(g1, g3, unit='m')
    result1 = "%.7f" % ret1
    result2 = "%.7f" % ret2
    return result1, result2

def cal_distance_ecef(row,baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    y1 = row['ecefY_gt']
    x1 = row['ecefX_gt']
    z1 = row['ecefZ_gt']
    y2 = row['Y_RLpredict']
    x2 = row['X_RLpredict']
    z2 = row['Z_RLpredict']
    if 'spp' in baseline_mod:
        y3 = row['YEcefMeters_spp']
        x3 = row['XEcefMeters_spp']
        z3 = row['ZEcefMeters_spp']
    elif baseline_mod == 'rtk':
        y3 = row['YEcefMeters_rtk']
        x3 = row['XEcefMeters_rtk']
        z3 = row['ZEcefMeters_rtk']
    #ret1 = haversine(g1, g2, unit='m')
    #ret2 = haversine(g1, g3, unit='m')
    llh1 = coord.ecef2geodetic([x1,y1,z1])
    llh3 = coord.ecef2geodetic([x3,y3,z3])
    llerr2 = haversine((llh1[0],llh1[1]), (llh3[0],llh3[1]), unit='m')
    # llerr2 = pmv.vdist(llh1[0],llh1[1], llh3[0],llh3[1])
    herr2 = (llh3[-1]-llh1[-1])
    herrabs2 = np.abs(llh3[-1]-llh1[-1])
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
        llh2 = coord.ecef2geodetic([x2,y2,z2])
        llerr1 = haversine((llh1[0],llh1[1]), (llh2[0],llh2[1]), unit='m')
        # llerr1 = pmv.vdist(llh1[0],llh1[1],llh2[0],llh2[1])
        herr1 = (llh2[-1]-llh1[-1])
        herrabs1 = np.abs(llh2[-1]-llh1[-1])
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
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)

def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score

def recording_results(data_truth_dic,trajdata_range,tripIDlist,logdirname):
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
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
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

def recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    rl_ll_mean_all = 0
    or_ll_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]
            if traj_record:
                # record rl traj
                if 'spp' in baseline_mod:
                    if 'XEcefMeters_ublox' in pd_train.columns:
                        record_columns = ['UTCtime', 'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict',
                                          'Z_RLpredict', 'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
                                          'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                                          'XEcefMeters_ublox', 'YEcefMeters_ublox', 'ZEcefMeters_ublox',
                                          'CN0_mean', 'EA_mean', 'PR_mean','satnum','velocity']  # UnixTimeMillis ,'HPL','2derr','HPL_gt','2derr_spp'
                    else:
                        record_columns=['UTCtime','ecefX_gt', 'ecefY_gt', 'ecefZ_gt','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict','XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
                                        'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                                        'CN0_mean','EA_mean','PR_mean','satnum','velocity']    # UnixTimeMillis ,'HPL','2derr','HPL_gt','2derr_spp'

                elif baseline_mod == 'rtk':
                    record_columns=['UTCtime','ecefX_gt', 'ecefY_gt', 'ecefZ_gt','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                                    'XEcefMeters_rtk', 'YEcefMeters_rtk', 'ZEcefMeters_rtk','CN0_mean','EA_mean','PR_mean','HPL']    # UnixTimeMillis
                try:
                    pd_record = pd_train[record_columns]
                except:
                    pd_record = pd_train[['UTCtime','ecefX_gt', 'ecefY_gt', 'ecefZ_gt','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict','XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
                                    'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']]
                # pd_record = pd_record[pd_record['X_RLpredict'].notnull()]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)

            if 'spp' in baseline_mod:
                test = pd_train.loc[:, ['ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict','XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                                        'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
            elif baseline_mod == 'rtk':
                test = pd_train.loc[:, ['ecefX_gt', 'ecefY_gt', 'ecefZ_gt','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_rtk', 'YEcefMeters_rtk', 'ZEcefMeters_rtk']]

            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[7], axis=1)
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
            rl_xdistance_mean=np.mean(test['rl_xdistance'])
            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
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
                           'or_habserr_max': np.nanmax(test['or_habsdistance'])
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

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
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
                           'or_habserr_max': np.nanmax(test['or_habsdistance'])
                           }
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            rl_ll_mean_all += test['rl_lldistance'].describe()['count'] * test['rl_lldistance'].describe()['mean']
            or_ll_mean_all += test['or_lldistance'].describe()['count'] * test['or_lldistance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except Exception as e:
            print(f'Trajectory {train_tripIDnum} error: {e}')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
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
    avg_xyz_err = rl_distances_mean_all / num_total_rl
    avg_xyz_or_err = or_distances_mean_all / num_total_rl
    avg_rl_llerr = rl_ll_mean_all / num_total_rl
    avg_or_llerr = or_ll_mean_all / num_total_rl

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')
    return avg_xyz_err,avg_xyz_or_err,avg_rl_llerr,avg_or_llerr