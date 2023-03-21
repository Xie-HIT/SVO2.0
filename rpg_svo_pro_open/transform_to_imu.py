import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path of Newer Coolege groundtruth in base coordinate')
    parser.add_argument('--output', type=str, required=True, help='path of Newer Coolege groundtruth in Alphasense IMU coordinate')
    args = parser.parse_args()

    pd_reader = pd.read_csv(args.input, delim_whitespace=True,
                            names=['时间戳', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'],
                            usecols=['时间戳', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'],
                            dtype={'时间戳': float, 'tx': float, 'ty': float, 'tz': float,
                                   'qx': float, 'qy': float, 'qz': float, 'qw': float},
                            skiprows=1)
    print(pd_reader.head(3))

    for i, row in pd_reader.iterrows():
        timestamp = row['时间戳']

        r1 = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])
        R1 = r1.as_matrix()
        T_w_imu = np.array([
            [R1[0][0], R1[0][1], R1[0][2], row['tx']],
            [R1[1][0], R1[1][1], R1[1][2], row['ty']],
            [R1[2][0], R1[2][1], R1[2][2], row['tz']],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        T_base_imu = np.array([
            [1.0, 0.0, 0.0, 0.038],
            [0.0, -1.0, 0.0, -0.008],
            [0.0, 0.0, -1.0, 0.065],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        T_imu_base = np.linalg.inv(T_base_imu)

        T_w_base = T_w_imu @ T_imu_base  # TODO (xie chen) 核心转换

        tx = T_w_base[0][3]
        ty = T_w_base[1][3]
        tz = T_w_base[2][3]
        r2 = R.from_matrix([
            [T_w_base[0][0], T_w_base[0][1], T_w_base[0][2]],
            [T_w_base[1][0], T_w_base[1][1], T_w_base[1][2]],
            [T_w_base[2][0], T_w_base[2][1], T_w_base[2][2]]
        ])
        q = r2.as_quat()  # [qx, qy, qz, qw]
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        new_row = pd.DataFrame([[timestamp, tx, ty, tz, qx, qy, qz, qw]], dtype=float)
        new_row.to_csv(args.output, sep=' ', header=False, index=False, mode='a')

    print('转换完成！\n')
