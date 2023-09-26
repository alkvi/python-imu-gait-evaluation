import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ahrs.filters import Madgwick
from ahrs import Quaternion
from ahrs import DEG2RAD, RAD2DEG
import utility_functions

fs_imu = 128

def extract_condition_data(imu_data, session, sensor_position, onset, duration):

    # Get session data
    seek_column = f"{session}/{sensor_position}/Accelerometer"
    acc_data = imu_data.filter(regex=seek_column).to_numpy()
    seek_column = f"{session}/{sensor_position}/Gyroscope"
    gyr_data = imu_data.filter(regex=seek_column).to_numpy()
    seek_column = f"{session}/{sensor_position}/Magnetometer"
    mag_data = imu_data.filter(regex=seek_column).to_numpy()

    # Get condition data. Get the closest sample to onset.
    time_axis = np.arange(0,acc_data.shape[0]) / fs_imu
    onset_idx = (np.abs(time_axis - onset)).argmin()
    end_idx = (np.abs(time_axis - (onset+duration))).argmin()

    # Cut to condition block
    acc_data = acc_data[onset_idx:end_idx,:]
    gyr_data = gyr_data[onset_idx:end_idx,:]
    mag_data = mag_data[onset_idx:end_idx,:]

    # Convert magnetometer data from microT to milliT
    mag_data = mag_data / 1000

    # For lumbar: Z should be X and reverse Y and X
    if sensor_position == "LUMBAR":
        acc_data = np.transpose(np.array([-acc_data[:,2], -acc_data[:,1], -acc_data[:,0]])) 
        gyr_data = np.transpose(np.array([-gyr_data[:,2], -gyr_data[:,1], -gyr_data[:,0]])) 
        mag_data = np.transpose(np.array([-mag_data[:,2], -mag_data[:,1], -mag_data[:,0]])) 

    return acc_data, gyr_data, mag_data

def get_start_quaternion(q_orig, direction, stable_sample = 500, plot_quaternions=False):
    
    # Convert to euler
    euler_angles = np.array([Quaternion(q_arr).to_angles() for q_arr in q_orig])
    euler_angles = np.degrees(euler_angles)

    # Get the mean stance angle in degrees
    mean_x = np.mean(euler_angles[stable_sample:-1,0])
    mean_y = np.mean(euler_angles[stable_sample:-1,1])
    mean_z = np.mean(euler_angles[stable_sample:-1,2])
    print(f"Euler mean x: {mean_x}, y: {mean_y}, z: {mean_z}")

    # We want to be facing 130 degrees south-east for forward.
    # Turn this around 180 degrees for the other direction.
    if direction == 'forward':
        start_z = 130
    else:
        start_z = -130
    euler_angles_modified = euler_angles.copy()
    euler_angles_modified[:,2] = start_z

    # Create a quaternion array from modified euler angles
    euler0 =np.flip(euler_angles_modified)
    q0 = np.array([Quaternion().from_angles(euler_arr*DEG2RAD) for euler_arr in euler0])

    # Plot the resulting quaternions
    if plot_quaternions:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ff_lf = [0]
        ff_rf = [0]
        utility_functions.plot_quaternions(ax1, q_orig, ff_lf, "Original quaternion")
        utility_functions.plot_quaternions(ax2, q0, ff_rf, "Rotated quaternion")
        utility_functions.plot_euler(ax3, q_orig, ff_lf, "Original euler")
        utility_functions.plot_euler(ax4, q0, ff_rf, "Rotated euler")
        plt.show()

    calibration_data = {
        'direction': [direction],
        'calibration_stance_x': [mean_x],
        'calibration_stance_y': [mean_y],
        'calibration_stance_z': [mean_z],
        'q0_w': [np.mean(q0[0:stable_sample,0])],
        'q0_x': [np.mean(q0[0:stable_sample,1])],
        'q0_y': [np.mean(q0[0:stable_sample,2])],
        'q0_z': [np.mean(q0[0:stable_sample,3])]
    }
    calibration_frame = pd.DataFrame(data=calibration_data)
    return calibration_frame

if __name__ == "__main__":

    # Where all the data is stored
    pq_folder = "../Data/imu_data_parquet"
    pq_files = [ f.path for f in os.scandir(pq_folder)]

    # Go through each pq file
    session = "calibration_2"
    all_subject_data = []
    for pq_file in pq_files:

        # Read IMU data
        imu_data = pd.read_parquet(pq_file)
        imu_columns = imu_data.columns
        subject = list(imu_columns)[0].split("/")[0]

        # If subject has no calibration data, skip. 
        # We'll give them a default start orientation later. 
        if session not in '\t'.join(imu_columns):
            print(f"{subject} has no calibration stance")
            continue

        print("Using pq file " + pq_file)
        acc_data_lf, gyro_data_lf, mag_data_lf = extract_condition_data(imu_data, session, "LEFT_FOOT", 0, 10)
        acc_data_rf, gyro_data_rf, mag_data_rf = extract_condition_data(imu_data, session, "RIGHT_FOOT", 0, 10)

        # Calculate orientation quaternions in a global frame (ENU) via Madgwick's orientation filter
        # Gain as recommended for MARG systems from Madgwick et al 2011
        madgwick_gain = 0.041
        madgwick_lf = Madgwick(gyr=gyro_data_lf, acc=acc_data_lf, mag=mag_data_lf, frequency=128, Dt=1/128, gain=madgwick_gain)
        madgwick_rf = Madgwick(gyr=gyro_data_rf, acc=acc_data_rf, mag=mag_data_rf, frequency=128, Dt=1/128, gain=madgwick_gain)

        subject_data = []
        calibration = get_start_quaternion(madgwick_lf.Q, 'forward')
        calibration.insert(0, "sensor", ["LEFT_FOOT"])
        calibration.insert(0, "subject", [subject])
        subject_data.append(calibration)
        calibration = get_start_quaternion(madgwick_lf.Q, 'backward')
        calibration.insert(0, "sensor", ["LEFT_FOOT"])
        calibration.insert(0, "subject", [subject])
        subject_data.append(calibration)
        calibration = get_start_quaternion(madgwick_rf.Q, 'forward')
        calibration.insert(0, "sensor", ["RIGHT_FOOT"])
        calibration.insert(0, "subject", [subject])
        subject_data.append(calibration)
        calibration = get_start_quaternion(madgwick_rf.Q, 'backward')
        calibration.insert(0, "sensor", ["RIGHT_FOOT"])
        calibration.insert(0, "subject", [subject])
        subject_data.append(calibration)
        subject_frame = pd.concat(subject_data)

        print(subject_frame)
        all_subject_data.append(subject_frame)

    all_subject_frame = pd.concat(all_subject_data)
    all_subject_frame.to_csv("../Data/calibration_stance_data.csv", index=False)
    print('done')