import numpy as np
from scipy.signal import filtfilt, butter
from matplotlib import pyplot as plt

# Prepare data numpy arrays
def prepare_data(data, lumbar):
    
    data_cols = [col for col in data.columns if 'Data' in col]

    acc_x_data = data[(data['Type'] == 'Accelerometer') & (
        data['Axis'] == 'x')][data_cols].dropna(axis=1).to_numpy().flatten()
    acc_y_data = data[(data['Type'] == 'Accelerometer') & (
        data['Axis'] == 'y')][data_cols].dropna(axis=1).to_numpy().flatten()
    acc_z_data = data[(data['Type'] == 'Accelerometer') & (
        data['Axis'] == 'z')][data_cols].dropna(axis=1).to_numpy().flatten()

    gyro_x_data = data[(data['Type'] == 'Gyroscope') & (
        data['Axis'] == 'x')][data_cols].dropna(axis=1).to_numpy().flatten()
    gyro_y_data = data[(data['Type'] == 'Gyroscope') & (
        data['Axis'] == 'y')][data_cols].dropna(axis=1).to_numpy().flatten()
    gyro_z_data = data[(data['Type'] == 'Gyroscope') & (
        data['Axis'] == 'z')][data_cols].dropna(axis=1).to_numpy().flatten()

    mag_x_data = data[(data['Type'] == 'Magnetometer') & (
        data['Axis'] == 'x')][data_cols].dropna(axis=1).to_numpy().flatten()
    mag_y_data = data[(data['Type'] == 'Magnetometer') & (
        data['Axis'] == 'y')][data_cols].dropna(axis=1).to_numpy().flatten()
    mag_z_data = data[(data['Type'] == 'Magnetometer') & (
        data['Axis'] == 'z')][data_cols].dropna(axis=1).to_numpy().flatten()

    q_1_data = data[(data['Type'] == 'Quaternion') & (
        data['Axis'] == '1')][data_cols].dropna(axis=1).to_numpy().flatten()
    q_2_data = data[(data['Type'] == 'Quaternion') & (
        data['Axis'] == '2')][data_cols].dropna(axis=1).to_numpy().flatten()
    q_3_data = data[(data['Type'] == 'Quaternion') & (
        data['Axis'] == '3')][data_cols].dropna(axis=1).to_numpy().flatten()
    q_4_data = data[(data['Type'] == 'Quaternion') & (
        data['Axis'] == '4')][data_cols].dropna(axis=1).to_numpy().flatten()
 
     # N-by-3 array
    acc_data = np.transpose(np.array([acc_x_data, acc_y_data, acc_z_data])) # in m/s^2
    gyro_data = np.transpose(np.array([gyro_x_data, gyro_y_data, gyro_z_data])) # in rad/s
    mag_data = np.transpose(np.array([mag_x_data/1000, mag_y_data/1000, mag_z_data/1000])) # convert from microT to milliT
    q_data = np.transpose(np.array([q_1_data, q_2_data, q_3_data, q_4_data])) # stored in (w,x,y,z)

    # For lumbar: Z should be X and reverse Y and X
    if lumbar:
        acc_data = np.transpose(np.array([-acc_z_data, -acc_y_data, -acc_x_data])) 
        gyro_data = np.transpose(np.array([-gyro_z_data, -gyro_y_data, -gyro_x_data])) 
        mag_data = np.transpose(np.array([-mag_z_data/1000, -mag_y_data/1000, -mag_x_data/1000])) 
        q_data = np.transpose(np.array([q_1_data, q_2_data, q_3_data, q_4_data]))
    
    return acc_data, gyro_data, mag_data, q_data

# Low-pass signals for smoothing, 4 Hz, 4th order butterworth
def filter_signal(gyro_data, acc_data, mag_data, fs, plot_filter):
    fn = fs/2
    b, a = butter(4, 4/fn, 'low')
    gyro_data_filt = filtfilt(b, a, gyro_data, axis=0)
    acc_data_filt = filtfilt(b, a, acc_data, axis=0)
    mag_data_filt = filtfilt(b, a, mag_data, axis=0)

    if plot_filter:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        t = np.arange(gyro_data.shape[0]) 
        ax1.plot(t, gyro_data[:,0], label='X orig')
        ax1.plot(t, gyro_data_filt[:,0], label='X filt')
        ax1.legend(loc="upper right")
        ax1.set_title('Gyro X filt')
        ax2.plot(t, gyro_data[:,1], label='Y orig')
        ax2.plot(t, gyro_data_filt[:,1], label='Y filt')
        ax2.legend(loc="upper right")
        ax2.set_title('Gyro Y filt')
        ax3.plot(t, gyro_data[:,2], label='Z orig')
        ax3.plot(t, gyro_data_filt[:,2], label='Z filt')
        ax3.legend(loc="upper right")
        ax3.set_title('Gyro Z filt')
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        t = np.arange(acc_data.shape[0]) 
        ax1.plot(t, acc_data[:,0], label='X orig')
        ax1.plot(t, acc_data_filt[:,0], label='X filt')
        ax1.legend(loc="upper right")
        ax1.set_title('Acc X filt')
        ax2.plot(t, acc_data[:,1], label='Y orig')
        ax2.plot(t, acc_data_filt[:,1], label='Y filt')
        ax2.legend(loc="upper right")
        ax2.set_title('Acc Y filt')
        ax3.plot(t, acc_data[:,2], label='Z orig')
        ax3.plot(t, acc_data_filt[:,2], label='Z filt')
        ax3.legend(loc="upper right")
        ax3.set_title('Acc Z filt')
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        t = np.arange(mag_data.shape[0]) 
        ax1.plot(t, mag_data[:,0], label='X orig')
        ax1.plot(t, mag_data_filt[:,0], label='X filt')
        ax1.legend(loc="upper right")
        ax1.set_title('Mag X filt')
        ax2.plot(t, mag_data[:,1], label='Y orig')
        ax2.plot(t, mag_data_filt[:,1], label='Y filt')
        ax2.legend(loc="upper right")
        ax2.set_title('Mag Y filt')
        ax3.plot(t, mag_data[:,2], label='Z orig')
        ax3.plot(t, mag_data_filt[:,2], label='Z filt')
        ax3.legend(loc="upper right")
        ax3.set_title('Mag Z filt')
        plt.show()

    return gyro_data_filt, acc_data_filt, mag_data_filt

# Cut signal to vicon events, with consideration to lag between signals
def cut_signal_to_events(vicon_event_data, xcorr_lag, data):

    # Get first and last event
    first_vicon_event = vicon_event_data.min().min()
    last_vicon_event = vicon_event_data.max().max()

    # Cut data to 1 second before first event, 4 seconds after last event
    cut_start = first_vicon_event - xcorr_lag - 1
    cut_end = last_vicon_event - xcorr_lag + 4
    cut_start_frame = int(np.floor(cut_start * 128))
    cut_end_frame = int(np.floor(cut_end * 128))
    if cut_end_frame > data.shape[0]:
        cut_end_frame = data.shape[0]
    if cut_start_frame < 0:
        cut_start_frame = 0
    data_cut = data[cut_start_frame:cut_end_frame,:]
    return data_cut, cut_start_frame

# Plot a quaternion
def plot_quaternions(ax, q, ff, title_str):
    t = np.arange(q.shape[0]) 
    ax.plot(t, q[:,0], label='w')
    ax.plot(t, q[:,1], label='x')
    ax.plot(t, q[:,2], label='y')
    ax.plot(t, q[:,3], label='z')
    ax.plot(ff, q[:,0][ff], 'k*')
    ax.plot(ff, q[:,1][ff], 'k*')
    ax.plot(ff, q[:,2][ff], 'k*')
    ax.plot(ff, q[:,3][ff], 'k*')
    ax.set_xlabel('Sample', fontsize = 15)
    ax.set_ylim(-1,1)
    ax.set_title(title_str, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.legend(loc="upper right")

# Plot two data with HS
def plot_axes_with_hs(data_1, data_2, hs_lf, hs_rf, label):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(data_1.shape[0]) 
    ax1.plot(t, data_1[:,0], label='X 1')
    ax1.plot(t, data_2[:,0], label='X 2')
    ax1.plot(hs_lf, data_2[:,0][hs_lf], 'r*', label='HS lf')
    ax1.plot(hs_rf, data_2[:,0][hs_rf], 'c*', label='HS rf')
    ax1.legend(loc="upper right")
    ax1.set_title(label + ' X')
    ax2.plot(t, data_1[:,1], label='Y 1')
    ax2.plot(t, data_2[:,1], label='Y 2')
    ax2.plot(hs_lf, data_2[:,1][hs_lf], 'r*', label='HS lf')
    ax2.plot(hs_rf, data_2[:,1][hs_rf], 'c*', label='HS rf')
    ax2.legend(loc="upper right")
    ax2.set_title(label + ' Y')
    ax3.plot(t, data_1[:,2], label='Z 1')
    ax3.plot(t, data_2[:,2], label='Z 2')
    ax3.plot(hs_lf, data_2[:,2][hs_lf], 'r*', label='HS lf')
    ax3.plot(hs_rf, data_2[:,2][hs_rf], 'c*', label='HS rf')
    ax3.legend(loc="upper right")
    ax3.set_title(label + ' Z')

# Helper function for plotting 3D
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

