# normalize parameters
# for 170 datasets lat [37.7770066, 34.1405127] lon [-118.315262, -122.4331823]
lat_min = 23.1419923966666
lat_max = 23.1848497849999
lon_min = 113.439967843333
lon_max = 113.52031935
# 79 trajs
xecef_normal = -2000000
yecef_normal = 5000000
zecef_normal = 2000000

xecef_min = -2344950
yecef_min = 5373925
zecef_min = 2471965
xecef_max = -2318450
yecef_max = 5393695
zecef_max = 2502528

# res_min=-200.55
# res_max=500.0
res_max = 1000000
losx_min = -0.975754418
losx_max = 0.621463432195474
losy_min = -0.229344003
losy_max = 0.987156913
losz_min = -0.718626583
losz_max = 0.878697739

# trajectory lengh
outlayer_in_end = 113
outlayer_in_end_ecef = 0

CN0_min = 20
CN0_max = 51
CN0_minmax = 15.800
CN0_maxmin = 45.400
CNO_distmin = 21.282
CNO_distmax = 46.283

PRU_min = 0.600
PRU_max = 149
PRU_minmax = 2.099
PRU_maxmin = 54.262
PRU_distmin = 0.897
PRU_distmax = 21.152

AA_min = 0.000
AA_max = 359.999
AA_minmax = 42.508
AA_maxmin = 317.554
AA_distmin = 24.960
AA_distmax = 338.317

EA_min = 0.262
EA_max = 1.2085
EA_minmax = 12.501
EA_maxmin = 67.262
EA_distmin = 7.675
EA_distmax = 68.257

Prate_min = -875.295
Prate_max = 952.739
Prate_minmax = -470.863
Prate_maxmin = 655.793
Prate_distmin = -558.267
Prate_distmax = 654.926

covx_max = 1.25
covx_min = -1.25
covv_max = 0.8
covv_min = -0.8

# define feature dimension
CN0PRUAAEA_num = 8  # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角+方位角
CN0PRUEA_num = 7  # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角
CN0EA_num = 6  # 伪距残差+LOS（3D）+CN0+高度角
CN0_num = 5  # 伪距残差+LOS（3D）+CN0
Prlos_num = 4
record_feature = True
# initial_P = 0.3
initial_P = 352.168283223011
initial_V = 248.444992818349
sigma_m_pos = 22.2014540146364
# sigma_m_vel = 3.16869196817363
sigma_m_vel = 3.16869196817363
outlier_velocity = 37  # m/s（国内最高限速33m/s）
interrupt_time = 5  # 中断时间
k_xy = 4.29
only0620 = False  # 只有0620的数据，这批数据有ublox
if only0620:
    sigma_v = 0.6 * 10
else:
    sigma_v = 3.16869196817363
sigma_x = 22.2014540146364  # default 12
sigma_a = 96.0931764293629
sigma_mahalanobis = 2000  # 异常测量检测
# 排除误差较大的canyon数据进行训练
exclude_canyon = ['20250620-PM_1-canyon1-0', '20250620-PM_2-canyon1-0', '20250701-AM-canyon1-0',
                  '20250701-AM-canyon3-0',
                  '20250703-AM_1-canyon1-0', '20250703-AM_2-canyon1-0', '20250708-AM_1-canyon1-0',
                  '20250708-AM_2-canyon1-0',
                  '20250708-AM_2-canyon1-1', '20250708-PM_1-canyon1-0', '20250711_1-canyon1-0', '20250711_1-canyon2-0',
                  '20250711_2-canyon1-0', '20250711_2-canyon2-0']
