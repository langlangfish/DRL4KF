# import library
import src.utils.gnss_lib.coordinates as coord
from src.utils.package import np, radians, sin, cos, asin, sqrt, haversine, pd, logging, os


def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0 - expFactor)
            J += expFactor
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
        J += expFactor
        rate = expFactor / J
        expRaw = (1 - rate) * expRaw
        expRaw += rate * data[j]
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


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
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


def cal_distance_ecef(row, baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    y1 = row['ecefY_gt']
    x1 = row['ecefX_gt']
    z1 = row['ecefZ_gt']
    y2 = row['Y_RLpredict']
    x2 = row['X_RLpredict']
    z2 = row['Z_RLpredict']
    x3, y3, z3 = None, None, None
    if 'spp' in baseline_mod:
        y3 = row['YEcefMeters_spp']
        x3 = row['XEcefMeters_spp']
        z3 = row['ZEcefMeters_spp']
    elif baseline_mod == 'rtk':
        y3 = row['YEcefMeters_rtk']
        x3 = row['XEcefMeters_rtk']
        z3 = row['ZEcefMeters_rtk']
    # ret1 = haversine(g1, g2, unit='m')
    # ret2 = haversine(g1, g3, unit='m')
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


def _process_test(test: pd.DataFrame, baseline_mod):
    # 创建所有距离量的列映射 (列名: 结果索引)
    distance_columns = {
        'rl_distance': 0,
        'or_distance': 1,
        'rl_xdistance': 2,
        'rl_ydistance': 3,
        'rl_zdistance': 4,
        'or_xdistance': 5,
        'or_ydistance': 6,
        'or_zdistance': 7,
        'rl_lldistance': 8,
        'rl_hdistance': 9,
        'or_lldistance': 10,
        'or_hdistance': 11,
        'rl_habsdistance': 12,
        'or_habsdistance': 13,
    }
    # 批量生成基础列
    results = test.apply(lambda row: cal_distance_ecef(row, baseline_mod), axis=1)
    for col, idx in distance_columns.items():
        test[col] = results.str[idx]

    # 生成误差列
    test['error'] = test['rl_distance'] - test['or_distance']
    # 批量生成count列
    column_lists = ['distance', 'xdistance', 'ydistance', 'zdistance', 'lldistance', 'hdistance',
                    'habsdistance']
    for prefix in ['rl', 'or']:
        for col in column_lists:
            test[f'count_{prefix}_{col}'] = test[f'{prefix}_{col}'].astype(float)
    return test


def gen_tmp_dic(train_tripIDnum: int, tripIDlist, test):
    # 基础键和统计量配置
    field_mapping = {
        # 格式：'原始字段后缀': '键名中间部分'
        'xdistance': 'xdistance',
        'ydistance': 'ydistance',
        'zdistance': 'zdistance',
        'lldistance': 'llerr',
        'hdistance': 'herr',
        'habsdistance': 'habserr',
    }

    stat_config = [
        ('mean', np.mean),
        ('std', np.std),
        ('min', np.nanmin),
        ('max', np.nanmax),
    ]

    tmp_dic = {'tripID': tripIDlist[train_tripIDnum]}

    # 批量生成统计字段
    for prefix in ['rl', 'or']:
        for data_field, key_field in field_mapping.items():
            col_name = f"{prefix}_{data_field}"
            for stat_name, stat_func in stat_config:
                tmp_dic[f"{prefix}_{key_field}_{stat_name}"] = stat_func(test[col_name])
    return tmp_dic


def _gen_test(pd_train: pd.DataFrame, baseline_mod: str):
    if 'spp' in baseline_mod:
        test = pd_train.loc[:, [
                                   'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                   'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                                   'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf'
                               ]]
    elif baseline_mod == 'rtk':
        test = pd_train.loc[:, [
                                   'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                   'XEcefMeters_rtk', 'YEcefMeters_rtk', 'ZEcefMeters_rtk'
                               ]]
    else:
        raise ValueError(f'test baseline_mod {baseline_mod} is not supported.')
    assert test is not None, f"test完成创建"
    return test


def traj_record(pd_train, baseline_mod, logdirname, tripIDlist, train_tripIDnum):
    if 'spp' in baseline_mod:
        if 'XEcefMeters_ublox' in pd_train.columns:
            record_columns = ['UTCtime', 'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict',
                              'Z_RLpredict', 'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
                              'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                              'XEcefMeters_ublox', 'YEcefMeters_ublox', 'ZEcefMeters_ublox',
                              'CN0_mean', 'EA_mean', 'PR_mean', 'satnum',
                              'VX_RLpredict','VY_RLpredict','VZ_RLpredict',
                              'VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']  # UnixTimeMillis ,'HPL','2derr','HPL_gt','2derr_spp'
        else:
            # 可选列：'CN0_mean','EA_mean','PR_mean','satnum','velocity''HPL', 'llerr_spp',
            #                               'llerr_rl',
            record_columns = ['UTCtime', 'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict',
                              'Z_RLpredict', 'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
                              'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                              'VX_RLpredict','VY_RLpredict','VZ_RLpredict',
                              'VXEcefMeters_wls','VYEcefMeters_wls','VZEcefMeters_wls']  # UnixTimeMillis ,

    elif baseline_mod == 'rtk':
        record_columns = ['UTCtime', 'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                          'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp',
                          'XEcefMeters_rtk', 'YEcefMeters_rtk', 'ZEcefMeters_rtk', 'CN0_mean', 'EA_mean', 'PR_mean',
                          'HPL']  # UnixTimeMillis
    else:
        raise ValueError(f'test baseline_mod {baseline_mod} is not supported.')
    try:
        pd_record = pd_train[record_columns]
    except Exception as e:
        logging.error(f"{e}")
        pd_record = pd_train[
            ['UTCtime', 'ecefX_gt', 'ecefY_gt', 'ecefZ_gt', 'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
             'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf',
             'XEcefMeters_spp', 'YEcefMeters_spp', 'ZEcefMeters_spp']]
    # pd_record = pd_record[pd_record['X_RLpredict'].notnull()]
    pd_record.to_csv(logdirname / f'rl_traj_{tripIDlist[train_tripIDnum].replace("/", "_")}.csv', index=True)


def recording_results_ecef(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                           baseline_mod,
                           _traj_record, verbose, step=None):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    rl_ll_mean_all = 0
    or_ll_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []
    # rl_x_err,rl_y_err,rl_z_err = [],[],[]
    # or_x_err, or_y_err, or_z_err = [], [], []
    for train_tripIDnum in eval_tripID_loop_range:
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            # print('当前记录结果的轨迹为：',tripIDlist[train_tripIDnum])
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]
            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname, tripIDlist=tripIDlist,
                            train_tripIDnum=train_tripIDnum)
            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)
            # rl_x_err.extend(test['rl_xdistance'].values)
            # rl_y_err.extend(test['rl_ydistance'].values)
            # rl_z_err.extend(test['rl_zdistance'].values)
            # or_x_err.extend(test['or_xdistance'].values)
            # or_y_err.extend(test['or_ydistance'].values)
            # or_z_err.extend(test['or_zdistance'].values)


            rl_lldistance = test['rl_lldistance']
            or_lldistance = test['or_lldistance']
            if verbose >= 2:
                logging.info(
                    f'RL LL distance: {np.mean(rl_lldistance): 4.3f} + {np.std(rl_lldistance): 4.3f}, OR LL distances: {np.mean(or_lldistance): 4.3f} + {np.std(or_lldistance): 4.3f}.')
            tmp_dic = gen_tmp_dic(train_tripIDnum=train_tripIDnum, tripIDlist=tripIDlist, test=test)
            if pd_gen:
                # 准备要拼接的新列
                new_error_col = test['error'].describe().to_frame(name=f'{train_tripIDnum}')
                new_error_col.loc['tripID'] = tripIDlist[train_tripIDnum]
                new_rl_distance_col = test['count_rl_distance'].describe().to_frame(name=f'{train_tripIDnum}')
                new_rl_distance_col.loc['tripID'] = tripIDlist[train_tripIDnum]
                new_or_distance_col = test['count_or_distance'].describe().to_frame(name=f'{train_tripIDnum}')
                new_or_distance_col.loc['tripID'] = tripIDlist[train_tripIDnum]
                new_xyz_distance_col = pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0].to_frame(
                    name=f'{train_tripIDnum}')

                # 使用 pd.concat 一次性拼接所有新列
                error_pd = pd.concat([error_pd, new_error_col], axis=1)
                rl_distance_pd = pd.concat([rl_distance_pd, new_rl_distance_col], axis=1)
                or_distance_pd = pd.concat([or_distance_pd, new_or_distance_col], axis=1)
                xyz_distance_pd = pd.concat([xyz_distance_pd, new_xyz_distance_col], axis=1)
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                error_pd.loc['tripID'] = tripIDlist[train_tripIDnum]
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                rl_distance_pd.loc['tripID'] = tripIDlist[train_tripIDnum]

                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'
                or_distance_pd.loc['tripID'] = tripIDlist[train_tripIDnum]

                xyz_distance_pd = pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen = True

            std_dic = {
                'error_std': test['error'].describe()['std'],
                'rl_distance_std': test['count_rl_distance'].describe()['std'],
                'or_distance_std': test['count_or_distance'].describe()['std'],
                'rl_ll_std': test['rl_lldistance'].describe()['std'],
                'or_ll_std': test['or_lldistance'].describe()['std'],
            }
            if len(test) == 1:  # 排除test只有一行无法计算std的极端情况
                for key, value in std_dic.items():
                    std_dic[key] = 0
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * \
                                     test['count_rl_distance'].describe()['mean']
            rl_ll_mean_all += test['rl_lldistance'].describe()['count'] * test['rl_lldistance'].describe()['mean']
            or_ll_mean_all += test['or_lldistance'].describe()['count'] * test['or_lldistance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * \
                                     test['count_or_distance'].describe()['mean']

            error_std_all += test['error'].describe()['count'] * std_dic['error_std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * \
                                    std_dic['rl_distance_std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * \
                                    std_dic['or_distance_std']
        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    assert error_pd is not None, "error_pd 不能为None"
    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    # 准备要拼接的新列数据
    avg_error_data = [
        num_total_err,
        error_mean_all / num_total_err,
        error_std_all / num_total_err,
        error_min,
        0,
        0,
        0,
        error_max,
        'AVG_ERR',
    ]
    avg_rl_distance_data = [
        num_total_rl,
        rl_distances_mean_all / num_total_rl,
        rl_distances_std_all / num_total_rl,
        np.min(rl_distance_pd.loc['min', :]),
        0,
        0,
        0,
        np.max(rl_distance_pd.loc['max', :]),
        'AVG_RL',
    ]
    avg_or_distance_data = [
        num_total_or,
        or_distances_mean_all / num_total_or,
        or_distances_std_all / num_total_or,
        np.min(or_distance_pd.loc['min', :]),
        0,
        0,
        0,
        np.max(or_distance_pd.loc['max', :]),
        'AVG_OR',
    ]

    # 创建新列的 Series，使用原始 DataFrame 的索引
    avg_error_series = pd.Series(data=avg_error_data, index=error_pd.index, name='Avg')
    avg_rl_distance_series = pd.Series(data=avg_rl_distance_data, index=rl_distance_pd.index, name='Avg')
    avg_or_distance_series = pd.Series(data=avg_or_distance_data, index=or_distance_pd.index, name='Avg')

    # 使用 pd.concat 一次性拼接所有新列
    error_pd = pd.concat([error_pd, avg_error_series], axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, avg_rl_distance_series], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, avg_or_distance_series], axis=1)
    os.makedirs(logdirname, exist_ok=True)
    if step is not None:
        error_pd.to_csv(logdirname / f'{test_type}_errors_step={step}.csv', index=True)
        rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances_step={step}.csv', index=True)
        or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances_step={step}.csv', index=True)
        xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_distances_step={step}.csv', index=True)
    else:
        error_pd.to_csv(logdirname / f'{test_type}_errors.csv', index=True)
        rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances.csv', index=True)
        or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances.csv', index=True)
        xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_distances.csv', index=True)
    avg_xyz_err = rl_distances_mean_all / num_total_rl
    avg_xyz_or_err = or_distances_mean_all / num_total_rl
    avg_rl_llerr = rl_ll_mean_all / num_total_rl
    avg_or_llerr = or_ll_mean_all / num_total_rl
    # calculate rmse of methods
    rl_distances_rmse = np.sqrt((np.array(rl_distances_rmse)**2).mean())
    or_distances_rmse = np.sqrt((np.array(or_distances_rmse)**2).mean())
    rl_llerr_rmse= np.sqrt((np.array(rl_llerr_rmse)**2).mean())
    or_llerr_rmse = np.sqrt((np.array(or_llerr_rmse)**2).mean())


    if verbose >= 2:
        logging.info(
            f'{test_type} Performances: count {num_total_err: 1.0f}, compared with baseline mean: {error_mean_all / num_total_err: 4.3f}+{error_std_all / num_total_err: 4.3f}m, '
            f'min: {error_min: 4.3f}m, max: {error_max: 4.3f}m.')
    result_dict = {
        'rl_3D_err': avg_xyz_err,
        'or_3D_err': avg_xyz_or_err,
        'rl_ll_err': avg_rl_llerr,
        'or_ll_err': avg_or_llerr,
        'rl_3D_err_RMSE': rl_distances_rmse,
        'or_3D_err_RMSE': or_distances_rmse,
        'rl_ll_err_RMSE': rl_llerr_rmse,
        'or_ll_err_RMSE': or_llerr_rmse,
    }
    return result_dict




def recording_results_ecef_xyz(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                           baseline_mod,
                           _traj_record, verbose, step=None):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    rl_ll_mean_all = 0
    or_ll_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    rl_xdistance_mean_all = 0
    rl_ydistance_mean_all = 0
    rl_zdistance_mean_all = 0
    or_xdistance_mean_all = 0
    or_ydistance_mean_all = 0
    or_zdistance_mean_all = 0
    rl_xdistance_std_all = 0
    rl_ydistance_std_all = 0
    rl_zdistance_std_all = 0
    or_xdistance_std_all = 0
    or_ydistance_std_all = 0
    or_zdistance_std_all = 0

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_xdistance_pd,rl_ydistance_pd,rl_zdistance_pd = None,None,None
    or_xdistance_pd, or_ydistance_pd, or_zdistance_pd = None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []

    # 循环开始
    for train_tripIDnum in eval_tripID_loop_range:
        try:
            current_trip_id = tripIDlist[train_tripIDnum]
            pd_train = data_truth_dic[current_trip_id]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]

            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname,
                            tripIDlist=tripIDlist, train_tripIDnum=train_tripIDnum)

            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)

            # --- 统计每个 trip 的 XYZ 轴 Mean 和 Std ---
            # 定义需要统计的列名映射
            # axes_cols = {
            #     'rl_x': 'rl_xdistance', 'rl_y': 'rl_ydistance', 'rl_z': 'rl_zdistance',
            #     'or_x': 'or_xdistance', 'or_y': 'or_ydistance', 'or_z': 'or_zdistance'
            # }
            #
            # trip_xyz_stats = {}
            # for label, col in axes_cols.items():
            #     if col in test.columns:
            #         trip_xyz_stats[f'{label}_mean'] = test[col].mean()
            #         trip_xyz_stats[f'{label}_std'] = test[col].std()
            #
            # trip_xyz_stats['tripID'] = current_trip_id
            # ---------------------------------------

            # 收集 RMSE 数据
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)

            # 更新汇总统计量 (用于最后计算 Avg)
            stats_desc = {
                'error': test['error'].describe(),
                'rl': test['count_rl_distance'].describe(),
                'or': test['count_or_distance'].describe(),
                'rl_ll': test['rl_lldistance'].describe(),
                'or_ll': test['or_lldistance'].describe(),
                'rl_x': test['rl_xdistance'].describe(),
                'rl_y': test['rl_xdistance'].describe(),
                'rl_z': test['rl_xdistance'].describe(),
                'or_x': test['or_xdistance'].describe(),
                'or_y': test['or_ydistance'].describe(),
                'or_z': test['or_zdistance'].describe(),
            }

            if pd_gen:
                # 拼接 Error, RL, OR 的 describe 数据
                error_pd = pd.concat([error_pd, stats_desc['error'].to_frame(name=f'{train_tripIDnum}')], axis=1)
                error_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                rl_distance_pd = pd.concat([rl_distance_pd, stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                or_distance_pd = pd.concat([or_distance_pd, stats_desc['or'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                rl_xdistance_pd = pd.concat([rl_xdistance_pd, stats_desc['rl_x'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_ydistance_pd = pd.concat([rl_ydistance_pd, stats_desc['rl_y'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_zdistance_pd = pd.concat([rl_zdistance_pd, stats_desc['rl_z'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_xdistance_pd = pd.concat([or_xdistance_pd, stats_desc['or_x'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_ydistance_pd = pd.concat([or_ydistance_pd, stats_desc['or_y'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_zdistance_pd = pd.concat([or_zdistance_pd, stats_desc['or_z'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)

                # 拼接 XYZ 详细统计数据
                # new_xyz_col = pd.Series(trip_xyz_stats, name=f'{train_tripIDnum}').to_frame()
                # xyz_distance_pd = pd.concat([xyz_distance_pd, new_xyz_col], axis=1)
            else:
                # 初始化
                error_pd = stats_desc['error'].to_frame(name=f'{train_tripIDnum}')
                error_pd.loc['tripID'] = current_trip_id

                rl_distance_pd = stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id

                or_distance_pd = stats_desc['or'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                rl_xdistance_pd = stats_desc['rl_x'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id

                rl_ydistance_pd = stats_desc['rl_y'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                rl_zdistance_pd = stats_desc['rl_z'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                or_xdistance_pd = stats_desc['or_x'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                or_ydistance_pd = stats_desc['or_y'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id
                or_zdistance_pd = stats_desc['or_z'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                # xyz_distance_pd = pd.Series(trip_xyz_stats, name=f'{train_tripIDnum}').to_frame()
                pd_gen = True

            # 累计用于计算全局平均值
            count = stats_desc['error']['count']
            error_mean_all += count * stats_desc['error']['mean']
            rl_distances_mean_all += count * stats_desc['rl']['mean']
            rl_ll_mean_all += count * stats_desc['rl_ll']['mean']
            or_ll_mean_all += count * stats_desc['or_ll']['mean']
            or_distances_mean_all += count * stats_desc['or']['mean']
            rl_xdistance_mean_all += count * stats_desc['rl_x']['mean']
            rl_ydistance_mean_all += count * stats_desc['rl_y']['mean']
            rl_zdistance_mean_all += count * stats_desc['rl_z']['mean']
            or_xdistance_mean_all += count * stats_desc['or_x']['mean']
            or_ydistance_mean_all += count * stats_desc['or_y']['mean']
            or_zdistance_mean_all += count * stats_desc['or_z']['mean']

            error_std_all += count * (stats_desc['error']['std'] if count > 1 else 0)
            rl_distances_std_all += count * (stats_desc['rl']['std'] if count > 1 else 0)
            or_distances_std_all += count * (stats_desc['or']['std'] if count > 1 else 0)
            rl_xdistance_std_all += count * (stats_desc['rl_x']['std'] if count > 1 else 0)
            rl_ydistance_std_all += count * (stats_desc['rl_y']['std'] if count > 1 else 0)
            rl_zdistance_std_all += count * (stats_desc['rl_z']['std'] if count > 1 else 0)
            or_xdistance_std_all += count * (stats_desc['or_x']['std'] if count > 1 else 0)
            or_ydistance_std_all += count * (stats_desc['or_y']['std'] if count > 1 else 0)
            or_zdistance_std_all += count * (stats_desc['or_z']['std'] if count > 1 else 0)

        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    # --- 循环结束后的汇总处理 ---
    assert error_pd is not None, "error_pd 不能为None"

    num_total_err = np.sum(error_pd.loc['count', :])

    # 构造 Avg 列
    def create_avg_series(pd_df, mean_val, std_val, tag):
        data = [num_total_err, mean_val, std_val, np.min(pd_df.loc['min', :]), 0, 0, 0, np.max(pd_df.loc['max', :]),
                tag]
        return pd.Series(data=data, index=pd_df.index, name='Avg')

    avg_error_series = create_avg_series(error_pd, error_mean_all / num_total_err, error_std_all / num_total_err,
                                         'AVG_ERR')
    avg_rl_series = create_avg_series(rl_distance_pd, rl_distances_mean_all / num_total_err,
                                      rl_distances_std_all / num_total_err, 'AVG_RL')
    avg_or_series = create_avg_series(or_distance_pd, or_distances_mean_all / num_total_err,
                                      or_distances_std_all / num_total_err, 'AVG_OR')
    avg_rlx_series = create_avg_series(rl_xdistance_pd, rl_xdistance_mean_all / num_total_err,
                                      rl_xdistance_std_all / num_total_err, 'AVG_xRL')
    avg_rly_series = create_avg_series(rl_ydistance_pd, rl_ydistance_mean_all / num_total_err,
                                      rl_ydistance_std_all / num_total_err, 'AVG_yRL')
    avg_rlz_series = create_avg_series(rl_zdistance_pd, rl_zdistance_mean_all / num_total_err,
                                      rl_zdistance_std_all / num_total_err, 'AVG_zRL')
    avg_orx_series = create_avg_series(or_xdistance_pd, or_xdistance_mean_all / num_total_err,
                                      or_xdistance_std_all / num_total_err, 'AVG_xor')
    avg_ory_series = create_avg_series(or_ydistance_pd, or_ydistance_mean_all / num_total_err,
                                      or_ydistance_std_all / num_total_err, 'AVG_yor')
    avg_orz_series = create_avg_series(or_zdistance_pd, or_zdistance_mean_all / num_total_err,
                                      or_zdistance_std_all / num_total_err, 'AVG_zor')

    error_pd = pd.concat([error_pd, avg_error_series], axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, avg_rl_series], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, avg_or_series], axis=1)
    rl_xdistance_pd = pd.concat([rl_xdistance_pd, avg_rlx_series], axis=1)
    rl_ydistance_pd = pd.concat([rl_ydistance_pd, avg_rly_series], axis=1)
    rl_zdistance_pd = pd.concat([rl_zdistance_pd, avg_rlz_series], axis=1)
    or_xdistance_pd = pd.concat([or_xdistance_pd, avg_orx_series], axis=1)
    or_ydistance_pd = pd.concat([or_ydistance_pd, avg_ory_series], axis=1)
    or_zdistance_pd = pd.concat([or_zdistance_pd, avg_orz_series], axis=1)

    # 保存文件
    os.makedirs(logdirname, exist_ok=True)
    suffix = f'_step={step}' if step is not None else ''
    error_pd.to_csv(logdirname / f'{test_type}_errors{suffix}.csv')
    rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances{suffix}.csv')
    or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances{suffix}.csv')
    xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_distances{suffix}.csv')
    rl_xdistance_pd.to_csv(logdirname / f'{test_type}_rl_xdistances{suffix}.csv')
    rl_ydistance_pd.to_csv(logdirname / f'{test_type}_rl_ydistances{suffix}.csv')
    rl_zdistance_pd.to_csv(logdirname / f'{test_type}_rl_zdistances{suffix}.csv')
    or_xdistance_pd.to_csv(logdirname / f'{test_type}_or_xdistances{suffix}.csv')
    or_ydistance_pd.to_csv(logdirname / f'{test_type}_or_ydistances{suffix}.csv')
    or_zdistance_pd.to_csv(logdirname / f'{test_type}_or_zdistances{suffix}.csv')
    # 计算 RMSE
    def calc_rmse(arr):
        return np.sqrt((np.array(arr) ** 2).mean())

    result_dict = {
        'rl_3D_err': rl_distances_mean_all / num_total_err,
        'or_3D_err': or_distances_mean_all / num_total_err,
        'rl_ll_err': rl_ll_mean_all / num_total_err,
        'or_ll_err': or_ll_mean_all / num_total_err,
        'rl_3D_err_RMSE': calc_rmse(rl_distances_rmse),
        'or_3D_err_RMSE': calc_rmse(or_distances_rmse),
        'rl_ll_err_RMSE': calc_rmse(rl_llerr_rmse),
        'or_ll_err_RMSE': calc_rmse(or_llerr_rmse),
    }

    if verbose >= 2:
        logging.info(f"{test_type} Done. Total Points: {num_total_err}")

    return result_dict


def recording_results_ecef_xyz_v2(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                               baseline_mod, _traj_record, verbose, step=None):
    # 基础累加变量
    error_mean_all, rl_distances_mean_all, or_distances_mean_all = 0, 0, 0
    rl_ll_mean_all, or_ll_mean_all = 0, 0
    error_std_all, rl_distances_std_all, or_distances_std_all = 0, 0, 0

    # XYZ 轴累加变量 (用于最后计算 Avg)
    xyz_keys = ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z']
    xyz_sums = {k: 0.0 for k in xyz_keys}
    xyz_stds_sum = {k: 0.0 for k in xyz_keys}

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []

    for train_tripIDnum in eval_tripID_loop_range:
        try:
            current_trip_id = tripIDlist[train_tripIDnum]
            pd_train = data_truth_dic[current_trip_id]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]

            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname,
                            tripIDlist=tripIDlist, train_tripIDnum=train_tripIDnum)

            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)

            # 收集 RMSE 原始数据
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)

            # 提取所有维度的描述性统计
            stats_desc = {
                'error': test['error'].describe(),
                'rl': test['count_rl_distance'].describe(),
                'or': test['count_or_distance'].describe(),
                'rl_ll': test['rl_lldistance'].describe(),
                'or_ll': test['or_lldistance'].describe(),
                'rl_x': test['rl_xdistance'].describe(),
                'rl_y': test['rl_ydistance'].describe(),
                'rl_z': test['rl_zdistance'].describe(),
                'or_x': test['or_xdistance'].describe(),
                'or_y': test['or_ydistance'].describe(),
                'or_z': test['or_zdistance'].describe(),
            }

            # 构造当前 Trip 的 XYZ 汇总 Series (Mean 和 Std 交替)
            xyz_series_list = []
            for k in xyz_keys:
                xyz_series_list.append(
                    pd.Series({f'{k}_mean': stats_desc[k]['mean'], f'{k}_std': stats_desc[k]['std']}))
            trip_xyz_summary = pd.concat(xyz_series_list)
            trip_xyz_summary['tripID'] = current_trip_id

            if pd_gen:
                error_pd = pd.concat([error_pd, stats_desc['error'].to_frame(name=f'{train_tripIDnum}')], axis=1)
                error_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                rl_distance_pd = pd.concat([rl_distance_pd, stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                or_distance_pd = pd.concat([or_distance_pd, stats_desc['or'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                # 拼接合并后的 XYZ 文件
                xyz_distance_pd = pd.concat([xyz_distance_pd, trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')],
                                            axis=1)
            else:
                error_pd = stats_desc['error'].to_frame(name=f'{train_tripIDnum}')
                error_pd.loc['tripID'] = current_trip_id
                rl_distance_pd = stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                or_distance_pd = stats_desc['or'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                xyz_distance_pd = trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')
                pd_gen = True

            # 累计计算全局 Avg
            count = stats_desc['error']['count']
            error_mean_all += count * stats_desc['error']['mean']
            rl_distances_mean_all += count * stats_desc['rl']['mean']
            rl_ll_mean_all += count * stats_desc['rl_ll']['mean']
            or_ll_mean_all += count * stats_desc['or_ll']['mean']
            or_distances_mean_all += count * stats_desc['or']['mean']

            error_std_all += count * (stats_desc['error']['std'] if count > 1 else 0)
            rl_distances_std_all += count * (stats_desc['rl']['std'] if count > 1 else 0)
            or_distances_std_all += count * (stats_desc['or']['std'] if count > 1 else 0)

            # XYZ 轴向累加
            for k in xyz_keys:
                xyz_sums[k] += count * stats_desc[k]['mean']
                xyz_stds_sum[k] += count * (stats_desc[k]['std'] if count > 1 else 0)

        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    # --- 循环结束后计算全局 Avg ---
    assert error_pd is not None, "error_pd 不能为None"
    num_total = np.sum(error_pd.loc['count', :])

    def create_avg_series(pd_df, mean_val, std_val, tag):
        # 参照 analysis 的 data 结构
        data = [num_total, mean_val, std_val, np.min(pd_df.loc['min', :]), 0, 0, 0, np.max(pd_df.loc['max', :]), tag]
        return pd.Series(data=data, index=pd_df.index, name='Avg')

    # 基础文件的 Avg 列
    error_pd = pd.concat(
        [error_pd, create_avg_series(error_pd, error_mean_all / num_total, error_std_all / num_total, 'AVG_ERR')],
        axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, create_avg_series(rl_distance_pd, rl_distances_mean_all / num_total,
                                                                  rl_distances_std_all / num_total, 'AVG_RL')], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, create_avg_series(or_distance_pd, or_distances_mean_all / num_total,
                                                                  or_distances_std_all / num_total, 'AVG_OR')], axis=1)

    # 构造 XYZ 文件的 Avg 列
    avg_xyz_dict = {}
    for k in xyz_keys:
        avg_xyz_dict[f'{k}_mean'] = xyz_sums[k] / num_total
        avg_xyz_dict[f'{k}_std'] = xyz_stds_sum[k] / num_total
    avg_xyz_dict['tripID'] = 'TOTAL_AVG'
    avg_xyz_series = pd.Series(avg_xyz_dict, index=xyz_distance_pd.index, name='Avg')
    xyz_distance_pd = pd.concat([xyz_distance_pd, avg_xyz_series], axis=1)

    # --- 保存文件 ---
    os.makedirs(logdirname, exist_ok=True)
    suffix = f'_step={step}' if step is not None else ''
    error_pd.to_csv(logdirname / f'{test_type}_errors{suffix}.csv')
    rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances{suffix}.csv')
    or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances{suffix}.csv')
    # 合并后的三轴文件
    xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_all_axes_stats{suffix}.csv')

    # 计算 RMSE
    def calc_rmse(arr):
        return np.sqrt((np.array(arr) ** 2).mean())

    result_dict = {
        'rl_3D_err': rl_distances_mean_all / num_total,
        'or_3D_err': or_distances_mean_all / num_total,
        'rl_ll_err': rl_ll_mean_all / num_total,
        'or_ll_err': or_ll_mean_all / num_total,
        'rl_3D_err_RMSE': calc_rmse(rl_distances_rmse),
        'or_3D_err_RMSE': calc_rmse(or_distances_rmse),
        'rl_ll_err_RMSE': calc_rmse(rl_llerr_rmse),
        'or_ll_err_RMSE': calc_rmse(or_llerr_rmse),
    }

    if verbose >= 2:
        logging.info(f"{test_type} Done. Total Points: {num_total}")

    return result_dict




def recording_results_ecef_xyz_v3(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                                  baseline_mod, _traj_record, verbose, step=None):
    # ================= 现有功能：基础累加变量 (保持不变) =================
    error_mean_all, rl_distances_mean_all, or_distances_mean_all = 0, 0, 0
    rl_ll_mean_all, or_ll_mean_all = 0, 0
    error_std_all, rl_distances_std_all, or_distances_std_all = 0, 0, 0

    xyz_keys = ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z']
    xyz_sums = {k: 0.0 for k in xyz_keys}
    xyz_stds_sum = {k: 0.0 for k in xyz_keys}

    # 【新增功能①】：建立一个空池子，用来收集当前类型的纯原始数据
    all_xyz_raw_data = {k: [] for k in xyz_keys}

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []

    for train_tripIDnum in eval_tripID_loop_range:
        try:
            current_trip_id = tripIDlist[train_tripIDnum]
            pd_train = data_truth_dic[current_trip_id]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]

            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname,
                            tripIDlist=tripIDlist, train_tripIDnum=train_tripIDnum)

            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)

            # 收集 RMSE 原始数据
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)

            # 【新增功能②】：提取当前单条轨迹的 XYZ 所有点的误差并放入池子
            all_xyz_raw_data['rl_x'].extend(test['rl_xdistance'].values)
            all_xyz_raw_data['rl_y'].extend(test['rl_ydistance'].values)
            all_xyz_raw_data['rl_z'].extend(test['rl_zdistance'].values)
            all_xyz_raw_data['or_x'].extend(test['or_xdistance'].values)
            all_xyz_raw_data['or_y'].extend(test['or_ydistance'].values)
            all_xyz_raw_data['or_z'].extend(test['or_zdistance'].values)

            # 提取所有维度的描述性统计 (保持不变)
            stats_desc = {
                'error': test['error'].describe(),
                'rl': test['count_rl_distance'].describe(),
                'or': test['count_or_distance'].describe(),
                'rl_ll': test['rl_lldistance'].describe(),
                'or_ll': test['or_lldistance'].describe(),
                'rl_x': test['rl_xdistance'].describe(),
                'rl_y': test['rl_ydistance'].describe(),
                'rl_z': test['rl_zdistance'].describe(),
                'or_x': test['or_xdistance'].describe(),
                'or_y': test['or_ydistance'].describe(),
                'or_z': test['or_zdistance'].describe(),
            }

            # 构造当前 Trip 的 XYZ 汇总 Series (保持不变)
            xyz_series_list = []
            for k in xyz_keys:
                xyz_series_list.append(
                    pd.Series({f'{k}_mean': stats_desc[k]['mean'], f'{k}_std': stats_desc[k]['std']}))
            trip_xyz_summary = pd.concat(xyz_series_list)
            trip_xyz_summary['tripID'] = current_trip_id

            if pd_gen:
                error_pd = pd.concat([error_pd, stats_desc['error'].to_frame(name=f'{train_tripIDnum}')], axis=1)
                error_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                rl_distance_pd = pd.concat([rl_distance_pd, stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                or_distance_pd = pd.concat([or_distance_pd, stats_desc['or'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                xyz_distance_pd = pd.concat([xyz_distance_pd, trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')],
                                            axis=1)
            else:
                error_pd = stats_desc['error'].to_frame(name=f'{train_tripIDnum}')
                error_pd.loc['tripID'] = current_trip_id
                rl_distance_pd = stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                or_distance_pd = stats_desc['or'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                xyz_distance_pd = trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')
                pd_gen = True

            # 累计计算全局 Avg (保持不变)
            count = stats_desc['error']['count']
            error_mean_all += count * stats_desc['error']['mean']
            rl_distances_mean_all += count * stats_desc['rl']['mean']
            rl_ll_mean_all += count * stats_desc['rl_ll']['mean']
            or_ll_mean_all += count * stats_desc['or_ll']['mean']
            or_distances_mean_all += count * stats_desc['or']['mean']

            error_std_all += count * (stats_desc['error']['std'] if count > 1 else 0)
            rl_distances_std_all += count * (stats_desc['rl']['std'] if count > 1 else 0)
            or_distances_std_all += count * (stats_desc['or']['std'] if count > 1 else 0)

            # XYZ 轴向累加 (保持不变)
            for k in xyz_keys:
                xyz_sums[k] += count * stats_desc[k]['mean']
                xyz_stds_sum[k] += count * (stats_desc[k]['std'] if count > 1 else 0)

        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    # --- 循环结束后计算全局 Avg (保持不变) ---
    assert error_pd is not None, "error_pd 不能为None"
    num_total = np.sum(error_pd.loc['count', :])

    def create_avg_series(pd_df, mean_val, std_val, tag):
        data = [num_total, mean_val, std_val, np.min(pd_df.loc['min', :]), 0, 0, 0, np.max(pd_df.loc['max', :]), tag]
        return pd.Series(data=data, index=pd_df.index, name='Avg')

    error_pd = pd.concat(
        [error_pd, create_avg_series(error_pd, error_mean_all / num_total, error_std_all / num_total, 'AVG_ERR')],
        axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, create_avg_series(rl_distance_pd, rl_distances_mean_all / num_total,
                                                                  rl_distances_std_all / num_total, 'AVG_RL')], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, create_avg_series(or_distance_pd, or_distances_mean_all / num_total,
                                                                  or_distances_std_all / num_total, 'AVG_OR')], axis=1)

    # 构造 XYZ 文件的 Avg 列 (保持不变)
    avg_xyz_dict = {}
    for k in xyz_keys:
        avg_xyz_dict[f'{k}_mean'] = xyz_sums[k] / num_total
        avg_xyz_dict[f'{k}_std'] = xyz_stds_sum[k] / num_total
    avg_xyz_dict['tripID'] = 'TOTAL_AVG'
    avg_xyz_series = pd.Series(avg_xyz_dict, index=xyz_distance_pd.index, name='Avg')
    xyz_distance_pd = pd.concat([xyz_distance_pd, avg_xyz_series], axis=1)

    # --- 保存现有文件 (保持不变) ---
    os.makedirs(logdirname, exist_ok=True)
    suffix = f'_step={step}' if step is not None else ''
    error_pd.to_csv(logdirname / f'{test_type}_errors{suffix}.csv')
    rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances{suffix}.csv')
    or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances{suffix}.csv')
    xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_all_axes_stats{suffix}.csv')

    # =================================================================================
    # 【新增功能③】：跨类型的全局总三轴平均值和 std (完全不影响原有逻辑)
    # =================================================================================
    try:
        global_raw_file = logdirname / f'GLOBAL_all_5types_raw_xyz_pool{suffix}.csv'

        # 1. 将当前 test_type 的所有点位原始误差，以"追加模式(mode='a')"存入一个公共文件
        current_type_df = pd.DataFrame(all_xyz_raw_data)
        if os.path.exists(global_raw_file):
            current_type_df.to_csv(global_raw_file, mode='a', header=False, index=False)
        else:
            current_type_df.to_csv(global_raw_file, mode='w', header=True, index=False)

        # 2. 读取这个不断积累的公共文件，计算当前所有类型加起来的真正总体 Mean 和 Std
        grand_df = pd.read_csv(global_raw_file)
        grand_stats = {}
        for k in xyz_keys:
            grand_stats[f'{k}_mean'] = grand_df[k].mean()
            grand_stats[f'{k}_std'] = grand_df[k].std()

        # 3. 每次都覆盖生成一个终极 Summary 文件。当你的 5 种类型跑完时，这个文件就是你要的最终结果！
        grand_stats_df = pd.DataFrame([grand_stats])
        grand_stats_df.to_csv(logdirname / f'FINAL_all_types_xyz_mean_std{suffix}.csv', index=False)

    except Exception as e:
        print(f"⚠️ 生成全局5类融合的三轴统计数据时出错: {e}")

    # =================================================================================

    # 计算 RMSE (保持不变)
    def calc_rmse(arr):
        return np.sqrt((np.array(arr) ** 2).mean())

    result_dict = {
        'rl_3D_err': rl_distances_mean_all / num_total,
        'or_3D_err': or_distances_mean_all / num_total,
        'rl_ll_err': rl_ll_mean_all / num_total,
        'or_ll_err': or_ll_mean_all / num_total,
        'rl_3D_err_RMSE': calc_rmse(rl_distances_rmse),
        'or_3D_err_RMSE': calc_rmse(or_distances_rmse),
        'rl_ll_err_RMSE': calc_rmse(rl_llerr_rmse),
        'or_ll_err_RMSE': calc_rmse(or_llerr_rmse),
    }

    if verbose >= 2:
        logging.info(f"{test_type} Done. Total Points: {num_total}")

    return result_dict


import os
import pandas as pd
import numpy as np
import logging


def recording_results_ecef_xyz_v4(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                                  baseline_mod, _traj_record, verbose, step=None):
    # 【提前定义 RMSE 计算函数，方便在循环中给每条轨迹单独算】
    def calc_rmse(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]  # 排除可能存在的 NaN
        if len(arr) == 0: return np.nan
        return np.sqrt((arr ** 2).mean())

    # ================= 现有功能：基础累加变量 (保持不变) =================
    error_mean_all, rl_distances_mean_all, or_distances_mean_all = 0, 0, 0
    rl_ll_mean_all, or_ll_mean_all = 0, 0
    error_std_all, rl_distances_std_all, or_distances_std_all = 0, 0, 0

    xyz_keys = ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z']
    xyz_sums = {k: 0.0 for k in xyz_keys}
    xyz_stds_sum = {k: 0.0 for k in xyz_keys}

    # ================= 全局数据池 (用于跨类型汇总) =================
    keys_to_extract = {
        'rl_x': 'rl_xdistance', 'rl_y': 'rl_ydistance', 'rl_z': 'rl_zdistance',
        'or_x': 'or_xdistance', 'or_y': 'or_ydistance', 'or_z': 'or_zdistance',
        'error': 'error',
        'rl_dist': 'count_rl_distance', 'or_dist': 'count_or_distance',
        'rl_ll': 'rl_lldistance', 'or_ll': 'or_lldistance',
        'rl_ll_rmse_data': 'count_rl_lldistance', 'or_ll_rmse_data': 'count_or_lldistance'
    }
    all_raw_data = {k: [] for k in keys_to_extract.keys()}

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []

    # 【新增】：用于记录当前类型下，每条独立轨迹的 RMSE，以此来算 mean 和 std
    trip_rl_3d_rmse_list = []
    trip_or_3d_rmse_list = []
    trip_rl_ll_rmse_list = []
    trip_or_ll_rmse_list = []

    for train_tripIDnum in eval_tripID_loop_range:
        try:
            current_trip_id = tripIDlist[train_tripIDnum]
            pd_train = data_truth_dic[current_trip_id]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]

            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname,
                            tripIDlist=tripIDlist, train_tripIDnum=train_tripIDnum)

            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)

            # 1. 收集用于计算该类型整体 RMSE 的原始数据
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)

            # 2. 【新增】：单独计算当前这条轨迹的 RMSE 并存入列表
            trip_rl_3d_rmse_list.append(calc_rmse(test['count_rl_distance'].values))
            trip_or_3d_rmse_list.append(calc_rmse(test['count_or_distance'].values))
            trip_rl_ll_rmse_list.append(calc_rmse(test['count_rl_lldistance'].values))
            trip_or_ll_rmse_list.append(calc_rmse(test['count_or_lldistance'].values))

            # 3. 把当前轨迹所有点位的所有误差指标全部倒进全局大池子
            for k, col in keys_to_extract.items():
                all_raw_data[k].extend(test[col].values)

            # 提取所有维度的描述性统计 (保持不变)
            stats_desc = {
                'error': test['error'].describe(),
                'rl': test['count_rl_distance'].describe(),
                'or': test['count_or_distance'].describe(),
                'rl_ll': test['rl_lldistance'].describe(),
                'or_ll': test['or_lldistance'].describe(),
                'rl_x': test['rl_xdistance'].describe(),
                'rl_y': test['rl_ydistance'].describe(),
                'rl_z': test['rl_zdistance'].describe(),
                'or_x': test['or_xdistance'].describe(),
                'or_y': test['or_ydistance'].describe(),
                'or_z': test['or_zdistance'].describe(),
            }

            # 构造当前 Trip 的 XYZ 汇总 Series (保持不变)
            xyz_series_list = []
            for k in xyz_keys:
                xyz_series_list.append(
                    pd.Series({f'{k}_mean': stats_desc[k]['mean'], f'{k}_std': stats_desc[k]['std']}))
            trip_xyz_summary = pd.concat(xyz_series_list)
            trip_xyz_summary['tripID'] = current_trip_id

            if pd_gen:
                error_pd = pd.concat([error_pd, stats_desc['error'].to_frame(name=f'{train_tripIDnum}')], axis=1)
                error_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                rl_distance_pd = pd.concat([rl_distance_pd, stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                or_distance_pd = pd.concat([or_distance_pd, stats_desc['or'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id

                xyz_distance_pd = pd.concat([xyz_distance_pd, trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')],
                                            axis=1)
            else:
                error_pd = stats_desc['error'].to_frame(name=f'{train_tripIDnum}')
                error_pd.loc['tripID'] = current_trip_id
                rl_distance_pd = stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                or_distance_pd = stats_desc['or'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id

                xyz_distance_pd = trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')
                pd_gen = True

            # 累计计算全局 Avg (保持不变)
            count = stats_desc['error']['count']
            error_mean_all += count * stats_desc['error']['mean']
            rl_distances_mean_all += count * stats_desc['rl']['mean']
            rl_ll_mean_all += count * stats_desc['rl_ll']['mean']
            or_ll_mean_all += count * stats_desc['or_ll']['mean']
            or_distances_mean_all += count * stats_desc['or']['mean']

            error_std_all += count * (stats_desc['error']['std'] if count > 1 else 0)
            rl_distances_std_all += count * (stats_desc['rl']['std'] if count > 1 else 0)
            or_distances_std_all += count * (stats_desc['or']['std'] if count > 1 else 0)

            # XYZ 轴向累加 (保持不变)
            for k in xyz_keys:
                xyz_sums[k] += count * stats_desc[k]['mean']
                xyz_stds_sum[k] += count * (stats_desc[k]['std'] if count > 1 else 0)

        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    # --- 循环结束后计算全局 Avg (保持不变) ---
    assert error_pd is not None, "error_pd 不能为None"
    num_total = np.sum(error_pd.loc['count', :])

    def create_avg_series(pd_df, mean_val, std_val, tag):
        data = [num_total, mean_val, std_val, np.min(pd_df.loc['min', :]), 0, 0, 0, np.max(pd_df.loc['max', :]), tag]
        return pd.Series(data=data, index=pd_df.index, name='Avg')

    error_pd = pd.concat(
        [error_pd, create_avg_series(error_pd, error_mean_all / num_total, error_std_all / num_total, 'AVG_ERR')],
        axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, create_avg_series(rl_distance_pd, rl_distances_mean_all / num_total,
                                                                  rl_distances_std_all / num_total, 'AVG_RL')], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, create_avg_series(or_distance_pd, or_distances_mean_all / num_total,
                                                                  or_distances_std_all / num_total, 'AVG_OR')], axis=1)

    # 构造 XYZ 文件的 Avg 列 (保持不变)
    avg_xyz_dict = {}
    for k in xyz_keys:
        avg_xyz_dict[f'{k}_mean'] = xyz_sums[k] / num_total
        avg_xyz_dict[f'{k}_std'] = xyz_stds_sum[k] / num_total
    avg_xyz_dict['tripID'] = 'TOTAL_AVG'
    avg_xyz_series = pd.Series(avg_xyz_dict, index=xyz_distance_pd.index, name='Avg')
    xyz_distance_pd = pd.concat([xyz_distance_pd, avg_xyz_series], axis=1)

    # --- 保存现有文件 (保持不变) ---
    os.makedirs(logdirname, exist_ok=True)
    suffix = f'_step={step}' if step is not None else ''
    error_pd.to_csv(logdirname / f'{test_type}_errors{suffix}.csv')
    rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances{suffix}.csv')
    or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances{suffix}.csv')
    xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_all_axes_stats{suffix}.csv')

    # =================================================================================
    # 【全局5类融合数据保存】 (保持不变)
    # =================================================================================
    try:
        global_raw_file = logdirname / f'GLOBAL_all_5types_raw_pool{suffix}.csv'
        current_type_df = pd.DataFrame(all_raw_data)
        if os.path.exists(global_raw_file):
            current_type_df.to_csv(global_raw_file, mode='a', header=False, index=False)
        else:
            current_type_df.to_csv(global_raw_file, mode='w', header=True, index=False)

        grand_df = pd.read_csv(global_raw_file)
        grand_stats = {}

        for k in ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z', 'error', 'rl_dist', 'or_dist', 'rl_ll', 'or_ll']:
            grand_stats[f'{k}_mean'] = grand_df[k].mean()
            grand_stats[f'{k}_std'] = grand_df[k].std()

        grand_stats['rl_3D_RMSE'] = calc_rmse(grand_df['rl_dist'])
        grand_stats['or_3D_RMSE'] = calc_rmse(grand_df['or_dist'])
        grand_stats['rl_ll_RMSE'] = calc_rmse(grand_df['rl_ll_rmse_data'])
        grand_stats['or_ll_RMSE'] = calc_rmse(grand_df['or_ll_rmse_data'])

        pd.DataFrame([grand_stats]).to_csv(logdirname / f'FINAL_all_types_global_metrics{suffix}.csv', index=False)
    except Exception as e:
        print(f"⚠️ 生成全局融合统计数据时出错: {e}")

    # =================================================================================
    # 【更新字典】：加入单类型的 RMSE Mean 和 Std
    # =================================================================================
    result_dict = {
        # 1. 常规的 Mean
        'rl_3D_err': rl_distances_mean_all / num_total,
        'or_3D_err': or_distances_mean_all / num_total,
        'rl_ll_err': rl_ll_mean_all / num_total,
        'or_ll_err': or_ll_mean_all / num_total,

        # 2. 整个当前类型所有点算出的 总 RMSE
        'rl_3D_err_RMSE': calc_rmse(rl_distances_rmse),
        'or_3D_err_RMSE': calc_rmse(or_distances_rmse),
        'rl_ll_err_RMSE': calc_rmse(rl_llerr_rmse),
        'or_ll_err_RMSE': calc_rmse(or_llerr_rmse),

        # 3. 【新加入】：当前类型下，各独立轨迹 RMSE 的 平均值 (Mean)
        'rl_3D_err_RMSE_mean': np.mean(trip_rl_3d_rmse_list),
        'or_3D_err_RMSE_mean': np.mean(trip_or_3d_rmse_list),
        'rl_ll_err_RMSE_mean': np.mean(trip_rl_ll_rmse_list),
        'or_ll_err_RMSE_mean': np.mean(trip_or_ll_rmse_list),

        # 4. 【新加入】：当前类型下，各独立轨迹 RMSE 的 标准差 (Std)
        'rl_3D_err_RMSE_std': np.std(trip_rl_3d_rmse_list),
        'or_3D_err_RMSE_std': np.std(trip_or_3d_rmse_list),
        'rl_ll_err_RMSE_std': np.std(trip_rl_ll_rmse_list),
        'or_ll_err_RMSE_std': np.std(trip_or_ll_rmse_list),
    }

    if verbose >= 2:
        logging.info(f"{test_type} Done. Total Points: {num_total}")
    summary_df = pd.DataFrame([result_dict])
    summary_df.to_csv(logdirname / f'{test_type}_summary_metrics{suffix}.csv', index=False)

    return result_dict


def recording_results_ecef_xyz_v5(test_type, data_truth_dic, eval_tripID_loop_range, tripIDlist, logdirname,
                                  baseline_mod, _traj_record, verbose, step=None):
    # 【提前定义 RMSE 计算函数，自动过滤可能存在的 NaN，保证算术准确】
    def calc_rmse(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return np.nan
        return np.sqrt((arr ** 2).mean())

    # ================= 1. 现有功能：基础累加变量 =================
    error_mean_all, rl_distances_mean_all, or_distances_mean_all = 0, 0, 0
    rl_ll_mean_all, or_ll_mean_all = 0, 0
    error_std_all, rl_distances_std_all, or_distances_std_all = 0, 0, 0

    xyz_keys = ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z']
    xyz_sums = {k: 0.0 for k in xyz_keys}
    xyz_stds_sum = {k: 0.0 for k in xyz_keys}

    # ================= 2. 全局数据池初始化 =================
    # 定义要提取的全部误差指标
    keys_to_extract = {
        'rl_x': 'rl_xdistance', 'rl_y': 'rl_ydistance', 'rl_z': 'rl_zdistance',
        'or_x': 'or_xdistance', 'or_y': 'or_ydistance', 'or_z': 'or_zdistance',
        'error': 'error',
        'rl_dist': 'count_rl_distance', 'or_dist': 'count_or_distance',
        'rl_ll': 'rl_lldistance', 'or_ll': 'or_lldistance',
        'rl_ll_rmse_data': 'count_rl_lldistance', 'or_ll_rmse_data': 'count_or_lldistance'
    }
    all_raw_data = {k: [] for k in keys_to_extract.keys()}

    pd_gen = False
    error_pd, rl_distance_pd, or_distance_pd, xyz_distance_pd = None, None, None, None
    rl_distances_rmse, rl_llerr_rmse = [], []
    or_distances_rmse, or_llerr_rmse = [], []

    # 用于专门记录当前类型下，每条独立轨迹的 RMSE 的列表
    trip_rl_3d_rmse_list = []
    trip_or_3d_rmse_list = []
    trip_rl_ll_rmse_list = []
    trip_or_ll_rmse_list = []

    for train_tripIDnum in eval_tripID_loop_range:
        try:
            current_trip_id = tripIDlist[train_tripIDnum]
            pd_train = data_truth_dic[current_trip_id]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]

            if _traj_record:
                traj_record(pd_train=pd_train, baseline_mod=baseline_mod, logdirname=logdirname,
                            tripIDlist=tripIDlist, train_tripIDnum=train_tripIDnum)

            test = _gen_test(pd_train=pd_train, baseline_mod=baseline_mod)
            test = _process_test(test=test, baseline_mod=baseline_mod)

            # 收集用于计算该类型整体 RMSE 的点级原始数据
            rl_distances_rmse.extend(test['count_rl_distance'].values)
            or_distances_rmse.extend(test['count_or_distance'].values)
            rl_llerr_rmse.extend(test['count_rl_lldistance'].values)
            or_llerr_rmse.extend(test['count_or_lldistance'].values)

            # 【单轨迹 RMSE 计算】：单独计算当前这条轨迹的 RMSE 并存入列表
            trip_rl_3d_rmse_list.append(calc_rmse(test['count_rl_distance'].values))
            trip_or_3d_rmse_list.append(calc_rmse(test['count_or_distance'].values))
            trip_rl_ll_rmse_list.append(calc_rmse(test['count_rl_lldistance'].values))
            trip_or_ll_rmse_list.append(calc_rmse(test['count_or_lldistance'].values))

            # 把当前轨迹所有点位的所有误差指标全部倒进全局大池子
            for k, col in keys_to_extract.items():
                all_raw_data[k].extend(test[col].values)

            # 提取描述性统计 (保持原样)
            stats_desc = {
                'error': test['error'].describe(),
                'rl': test['count_rl_distance'].describe(),
                'or': test['count_or_distance'].describe(),
                'rl_ll': test['rl_lldistance'].describe(),
                'or_ll': test['or_lldistance'].describe(),
                'rl_x': test['rl_xdistance'].describe(),
                'rl_y': test['rl_ydistance'].describe(),
                'rl_z': test['rl_zdistance'].describe(),
                'or_x': test['or_xdistance'].describe(),
                'or_y': test['or_ydistance'].describe(),
                'or_z': test['or_zdistance'].describe(),
            }

            # 构造当前 Trip 的 XYZ 汇总 Series (保持原样)
            xyz_series_list = []
            for k in xyz_keys:
                xyz_series_list.append(
                    pd.Series({f'{k}_mean': stats_desc[k]['mean'], f'{k}_std': stats_desc[k]['std']}))
            trip_xyz_summary = pd.concat(xyz_series_list)
            trip_xyz_summary['tripID'] = current_trip_id

            if pd_gen:
                error_pd = pd.concat([error_pd, stats_desc['error'].to_frame(name=f'{train_tripIDnum}')], axis=1)
                error_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                rl_distance_pd = pd.concat([rl_distance_pd, stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                rl_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                or_distance_pd = pd.concat([or_distance_pd, stats_desc['or'].to_frame(name=f'{train_tripIDnum}')],
                                           axis=1)
                or_distance_pd.loc['tripID', f'{train_tripIDnum}'] = current_trip_id
                xyz_distance_pd = pd.concat([xyz_distance_pd, trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')],
                                            axis=1)
            else:
                error_pd = stats_desc['error'].to_frame(name=f'{train_tripIDnum}')
                error_pd.loc['tripID'] = current_trip_id
                rl_distance_pd = stats_desc['rl'].to_frame(name=f'{train_tripIDnum}')
                rl_distance_pd.loc['tripID'] = current_trip_id
                or_distance_pd = stats_desc['or'].to_frame(name=f'{train_tripIDnum}')
                or_distance_pd.loc['tripID'] = current_trip_id
                xyz_distance_pd = trip_xyz_summary.to_frame(name=f'{train_tripIDnum}')
                pd_gen = True

            # 累计计算局部 Avg (保持原样)
            count = stats_desc['error']['count']
            error_mean_all += count * stats_desc['error']['mean']
            rl_distances_mean_all += count * stats_desc['rl']['mean']
            rl_ll_mean_all += count * stats_desc['rl_ll']['mean']
            or_ll_mean_all += count * stats_desc['or_ll']['mean']
            or_distances_mean_all += count * stats_desc['or']['mean']

            error_std_all += count * (stats_desc['error']['std'] if count > 1 else 0)
            rl_distances_std_all += count * (stats_desc['rl']['std'] if count > 1 else 0)
            or_distances_std_all += count * (stats_desc['or']['std'] if count > 1 else 0)

            for k in xyz_keys:
                xyz_sums[k] += count * stats_desc[k]['mean']
                xyz_stds_sum[k] += count * (stats_desc[k]['std'] if count > 1 else 0)

        except Exception as e:
            print(f'Episode {train_tripIDnum} error: {e}')

    # ================= 3. 计算本类型的汇总表格并保存 =================
    assert error_pd is not None, "error_pd 不能为None"
    num_total = np.sum(error_pd.loc['count', :])

    def create_avg_series(pd_df, mean_val, std_val, tag):
        data = [num_total, mean_val, std_val, np.min(pd_df.loc['min', :]), 0, 0, 0, np.max(pd_df.loc['max', :]), tag]
        return pd.Series(data=data, index=pd_df.index, name='Avg')

    error_pd = pd.concat(
        [error_pd, create_avg_series(error_pd, error_mean_all / num_total, error_std_all / num_total, 'AVG_ERR')],
        axis=1)
    rl_distance_pd = pd.concat([rl_distance_pd, create_avg_series(rl_distance_pd, rl_distances_mean_all / num_total,
                                                                  rl_distances_std_all / num_total, 'AVG_RL')], axis=1)
    or_distance_pd = pd.concat([or_distance_pd, create_avg_series(or_distance_pd, or_distances_mean_all / num_total,
                                                                  or_distances_std_all / num_total, 'AVG_OR')], axis=1)

    avg_xyz_dict = {}
    for k in xyz_keys:
        avg_xyz_dict[f'{k}_mean'] = xyz_sums[k] / num_total
        avg_xyz_dict[f'{k}_std'] = xyz_stds_sum[k] / num_total
    avg_xyz_dict['tripID'] = 'TOTAL_AVG'
    avg_xyz_series = pd.Series(avg_xyz_dict, index=xyz_distance_pd.index, name='Avg')
    xyz_distance_pd = pd.concat([xyz_distance_pd, avg_xyz_series], axis=1)

    os.makedirs(logdirname, exist_ok=True)
    suffix = f'_step={step}' if step is not None else ''
    error_pd.to_csv(logdirname / f'{test_type}_errors{suffix}.csv')
    rl_distance_pd.to_csv(logdirname / f'{test_type}_rl_distances{suffix}.csv')
    or_distance_pd.to_csv(logdirname / f'{test_type}_or_distances{suffix}.csv')
    xyz_distance_pd.to_csv(logdirname / f'{test_type}_xyz_all_axes_stats{suffix}.csv')

    # ================= 4. 全局跨类型统计池子计算 (追加写入模式) =================
    try:
        # --- (A) 点级别原始数据大池子 ---
        global_raw_file = logdirname / f'GLOBAL_all_5types_raw_pool{suffix}.csv'
        current_type_df = pd.DataFrame(all_raw_data)
        if os.path.exists(global_raw_file):
            current_type_df.to_csv(global_raw_file, mode='a', header=False, index=False)
        else:
            current_type_df.to_csv(global_raw_file, mode='w', header=True, index=False)

        grand_df = pd.read_csv(global_raw_file)
        grand_stats = {}

        # 计算点级别的跨类型全局 Mean 和 Std
        for k in ['rl_x', 'rl_y', 'rl_z', 'or_x', 'or_y', 'or_z', 'error', 'rl_dist', 'or_dist', 'rl_ll', 'or_ll']:
            grand_stats[f'{k}_mean'] = grand_df[k].mean()
            grand_stats[f'{k}_std'] = grand_df[k].std()

        # 计算点级别的跨类型全局整体 RMSE
        grand_stats['rl_3D_RMSE'] = calc_rmse(grand_df['rl_dist'])
        grand_stats['or_3D_RMSE'] = calc_rmse(grand_df['or_dist'])
        grand_stats['rl_ll_RMSE'] = calc_rmse(grand_df['rl_ll_rmse_data'])
        grand_stats['or_ll_RMSE'] = calc_rmse(grand_df['or_ll_rmse_data'])

        # --- (B) 轨迹级别 RMSE 大池子 ---
        global_trip_rmse_file = logdirname / f'GLOBAL_all_5types_trip_rmse_pool{suffix}.csv'
        current_type_trip_rmses = pd.DataFrame({
            'trip_rl_3D_RMSE': trip_rl_3d_rmse_list,
            'trip_or_3D_RMSE': trip_or_3d_rmse_list,
            'trip_rl_ll_RMSE': trip_rl_ll_rmse_list,
            'trip_or_ll_RMSE': trip_or_ll_rmse_list
        })

        if os.path.exists(global_trip_rmse_file):
            current_type_trip_rmses.to_csv(global_trip_rmse_file, mode='a', header=False, index=False)
        else:
            current_type_trip_rmses.to_csv(global_trip_rmse_file, mode='w', header=True, index=False)

        grand_trip_rmse_df = pd.read_csv(global_trip_rmse_file)

        # 计算所有 5 种类型累计几十条轨迹的 RMSE 全局 Mean 和 Std
        for col in ['trip_rl_3D_RMSE', 'trip_or_3D_RMSE', 'trip_rl_ll_RMSE', 'trip_or_ll_RMSE']:
            grand_stats[f'GLOBAL_{col}_mean'] = grand_trip_rmse_df[col].mean()
            grand_stats[f'GLOBAL_{col}_std'] = grand_trip_rmse_df[col].std()

        # 保存这 5 个类型融合的终极大表
        grand_stats_df = pd.DataFrame([grand_stats])
        grand_stats_df.to_csv(logdirname / f'FINAL_all_types_global_metrics{suffix}.csv', index=False)

    except Exception as e:
        print(f"⚠️ 生成全局融合统计数据时出错: {e}")

    # ================= 5. 构建当前类型的返回字典与单独保存 =================
    result_dict = {
        # 1. 常规 Mean
        'rl_3D_err': rl_distances_mean_all / num_total,
        'or_3D_err': or_distances_mean_all / num_total,
        'rl_ll_err': rl_ll_mean_all / num_total,
        'or_ll_err': or_ll_mean_all / num_total,

        # 2. 当前类型的 点级别整体 RMSE
        'rl_3D_err_RMSE': calc_rmse(rl_distances_rmse),
        'or_3D_err_RMSE': calc_rmse(or_distances_rmse),
        'rl_ll_err_RMSE': calc_rmse(rl_llerr_rmse),
        'or_ll_err_RMSE': calc_rmse(or_llerr_rmse),

        # 3. 当前类型下，各独立轨迹 RMSE 的 Mean
        'rl_3D_err_RMSE_mean': np.mean(trip_rl_3d_rmse_list),
        'or_3D_err_RMSE_mean': np.mean(trip_or_3d_rmse_list),
        'rl_ll_err_RMSE_mean': np.mean(trip_rl_ll_rmse_list),
        'or_ll_err_RMSE_mean': np.mean(trip_or_ll_rmse_list),

        # 4. 当前类型下，各独立轨迹 RMSE 的 Std
        'rl_3D_err_RMSE_std': np.std(trip_rl_3d_rmse_list),
        'or_3D_err_RMSE_std': np.std(trip_or_3d_rmse_list),
        'rl_ll_err_RMSE_std': np.std(trip_rl_ll_rmse_list),
        'or_ll_err_RMSE_std': np.std(trip_or_ll_rmse_list),
    }

    # 【直接将上述包含 16 个核心指标的字典落盘，一目了然！】
    summary_df = pd.DataFrame([result_dict])
    summary_df.to_csv(logdirname / f'{test_type}_summary_metrics{suffix}.csv', index=False)

    if verbose >= 2:
        logging.info(f"{test_type} Done. Total Points: {num_total}")

    return result_dict