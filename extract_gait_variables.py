import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utility_functions
import gait_event_detection
import position_estimation
import gait_parameter_estimation

fs_imu = 128

# Class containing gait parameters for a bout of walking
class SegmentParameters:

    # Constructor
    def __init__(self, means_data, step_times_left, step_times_right, stride_lengths_left, stride_lengths_right):
        self.means_data = means_data
        self.step_times_left = step_times_left
        self.step_times_right = step_times_right
        self.stride_lengths_left = stride_lengths_left
        self.stride_lengths_right = stride_lengths_right
        self.trial_type = "None"

    # Factory function to initialize from multiple SegmentParameters
    # This will take the average of all segments for means data, and 
    # concatenate all vector data
    @classmethod
    def from_multiple_segments(cls, segments):

        # If we have no segments coming in, create empty object
        if len(segments) < 1:
            return cls(None, [], [], [], [])
        
        # Get the means
        segment_means = [segment.means_data for segment in segments]
        means_frame = pd.DataFrame([pd.concat(segment_means).mean(axis=0)])
        
        # Concatenate vector data
        step_times_left = np.concatenate([params.step_times_left for params in segments])
        step_times_right = np.concatenate([params.step_times_right for params in segments])
        stride_lengths_left = np.concatenate([params.stride_lengths_left for params in segments])
        stride_lengths_right = np.concatenate([params.stride_lengths_right for params in segments])

        # Create data class
        return cls(means_frame, step_times_left, step_times_right, stride_lengths_left, stride_lengths_right)
    
    # Copy from another SegmentParameters
    def copy(self, other):
        self.means_data = other.means_data
        self.step_times_left = other.step_times_left
        self.step_times_right = other.step_times_right
        self.stride_lengths_left = other.stride_lengths_left
        self.stride_lengths_right = other.stride_lengths_right

    # Set trial type
    def set_trial_type(self, trial_type):
        self.trial_type = trial_type

# Calculate variability parameters according to Galna et al 2013
def calculate_variability_parameters(block_parameters, subject, session):

    # Group by trial type
    trial_types = np.unique([params.trial_type for params in block_parameters])
    all_frames = []
    for trial in trial_types:
        trial_block_parameters = [params for params in block_parameters if params.trial_type == trial]
        trial_parameters = SegmentParameters.from_multiple_segments(trial_block_parameters)
        
        # Variability calculation
        step_time_sd_lr = np.sqrt(
            (np.var(trial_parameters.step_times_left) + 
            np.var(trial_parameters.step_times_right)) / 2)
        stride_length_sd_lr = np.sqrt(
            (np.var(trial_parameters.stride_lengths_left) + 
            np.var(trial_parameters.stride_lengths_right)) / 2)
        
        # For asymmetry: take difference of each gait cycle
        # e.g. step_times_left[0] - step_times_right[0]
        # Asymmetry step time
        step_time_min_length = np.min([len(trial_parameters.step_times_left), len(trial_parameters.step_times_right)])
        lr_array = np.array([np.array(trial_parameters.step_times_left[0:step_time_min_length]).astype(float),
                             np.array(trial_parameters.step_times_right[0:step_time_min_length]).astype(float)]).astype(float)
        mean_array = np.mean(lr_array, axis=0)
        step_time_diff_array = np.abs(np.diff(lr_array, axis=0))
        step_time_diff_array_percent = np.divide(step_time_diff_array, mean_array)*100

        # Asymmetry stride length
        stride_length_min_length = np.min([len(trial_parameters.stride_lengths_left), len(trial_parameters.stride_lengths_right)])
        lr_array = np.array([np.array(trial_parameters.stride_lengths_left[0:stride_length_min_length]).astype(float),
                             np.array(trial_parameters.stride_lengths_right[0:stride_length_min_length]).astype(float)]).astype(float)
        mean_array = np.mean(lr_array, axis=0)
        stride_length_diff_array = np.abs(np.diff(lr_array, axis=0))
        stride_length_diff_array_percent = np.divide(stride_length_diff_array, mean_array)*100

        # Compile parameters
        var_data = {
            'subject': [subject],
            'session': [session],
            'trial_type': [trial],
            'Step Time Variability': [step_time_sd_lr],
            'Step Time Variability Amount Used Values (L+R)': [len(trial_parameters.step_times_left) + len(trial_parameters.step_times_right)],
            'Stride Length Variability': [stride_length_sd_lr],
            'Stride Length Variability Amount Used Values (L+R)': [len(trial_parameters.stride_lengths_left) + len(trial_parameters.stride_lengths_right)],
            'Step Time Asymmetry': [np.mean(step_time_diff_array)],
            'Step Time Asymmetry Percent': [np.mean(step_time_diff_array_percent)],
            'Step Time Asymmetry Amount Used Values': [step_time_min_length],
            'Stride Length Asymmetry': [np.mean(stride_length_diff_array)],
            'Stride Length Asymmetry Percent': [np.mean(stride_length_diff_array_percent)],
            'Stride Length Asymmetry Amount Used Values': [stride_length_min_length],
        }
        var_frame = pd.DataFrame(data=var_data)
        all_frames.append(var_frame)
    
    session_var_frame = pd.concat(all_frames)
    return session_var_frame


def extract_turn_times(lumbar_gyro_data, mag_data_lumbar, plot_turn_regions=0, min_turn_limit=2.0, min_region_length=30):
    
    # Consider a turning period where abs(gyro) > limit, around vertical (Z) axis of lumbar.
    turn_idx = np.argwhere(np.abs(lumbar_gyro_data[:,2]) > min_turn_limit).astype(int)
    
    # Get contiguous regions
    index_diffs = np.ediff1d(turn_idx)
    region_boundary_idx = np.argwhere(index_diffs > 1).astype(int)
    if len(region_boundary_idx) > 0:
        # We have several regions, extract regions into a list
        region_boundary_idx = region_boundary_idx[:,0]
        turn_regions = np.split(turn_idx, region_boundary_idx+1, axis=0)
    else:
        # Put the single region into a list
        turn_regions = [turn_idx]

    # Keep only those regions above a certain length
    final_turn_regions = np.array([region for region in turn_regions if len(region) > min_region_length], dtype=object)

    # Plot turn regions
    if plot_turn_regions:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(mag_data_lumbar)
        ax2.plot(lumbar_gyro_data[:,2])
        ax1.legend(['x','y','z'])
        if len(final_turn_regions) > 0:
            for region in final_turn_regions:
                print(f"Start: {region[0]}, end: {region[-1]}")
                try:
                    ax1.axhline(0, linestyle='--', color='b', alpha=0.5)
                    ax1.axhline(np.mean(mag_data_lumbar[0:int(region[0]),1]), linestyle='--', color='r', alpha=0.5)
                    ax1.axhline(np.mean(mag_data_lumbar[int(region[-1]):-1,1]), linestyle='--', color='r', alpha=0.5)
                    ax1.axvspan(region[0], region[-1], facecolor="red", alpha=0.2, zorder=-100)
                    ax2.axhline(min_turn_limit, linestyle='--', color='r', alpha=0.5)
                    ax2.axvspan(region[0], region[-1], facecolor="red", alpha=0.2, zorder=-100)
                except:
                    print("Could not plot turns")
        plt.show()

    return final_turn_regions


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

def get_segment_data(acc_data, gyr_data, mag_data, segment_start, segment_end):
    acc_data = acc_data[segment_start:segment_end,:]
    gyr_data = gyr_data[segment_start:segment_end,:]
    mag_data = mag_data[segment_start:segment_end,:]
    return acc_data, gyr_data, mag_data

def extract_straight_walking(data_lf, data_rf, ff_lf, ff_rf, hs_lf, hs_rf, to_lf, to_rf, direction, calibration_data):
    
    # Approximate default orientation if no calibration data
    if direction == 'forward':
        q0_default = [0.256, -0.320, 0.149, 0.900]
    else:
        q0_default = [0.256, 0.149, -0.320, -0.900]

    # Set initial orientation
    calibration_data_dir = calibration_data[calibration_data['direction'] == direction]
    if len(calibration_data_dir) < 1:
        q0_lf = q0_default
        q0_rf = q0_default
        print(f"Using default q0 {q0_default}")
    else:
        calibration_data_dir_lf = calibration_data_dir[calibration_data_dir['sensor'] == "LEFT_FOOT"]
        calibration_data_dir_rf = calibration_data_dir[calibration_data_dir['sensor'] == "RIGHT_FOOT"]
        q0_lf = [calibration_data_dir_lf['q0_w'].values[0], calibration_data_dir_lf['q0_x'].values[0], 
                 calibration_data_dir_lf['q0_y'].values[0], calibration_data_dir_lf['q0_z'].values[0]]
        q0_rf = [calibration_data_dir_rf['q0_w'].values[0], calibration_data_dir_rf['q0_x'].values[0], 
                 calibration_data_dir_rf['q0_y'].values[0], calibration_data_dir_rf['q0_z'].values[0]]
        print(f"Using q0 LF {q0_lf}")
        print(f"Using q0 RF {q0_rf}")

    # Get the positions of sensors at each time point in the global frame
    positions_lf = position_estimation.get_positions_global_per_ff(data_lf, q0_lf, ff_lf, plot_positions, None)
    positions_rf = position_estimation.get_positions_global_per_ff(data_rf, q0_rf, ff_rf, plot_positions, None)

    # Calculate stride length and walking speed
    stride_lengths_lf, walking_speeds_lf =  gait_parameter_estimation.get_stride_length_walking_speed_foot(ff_lf, positions_lf, fs_imu, plot_trajectory)
    stride_lengths_rf, walking_speeds_rf =  gait_parameter_estimation.get_stride_length_walking_speed_foot(ff_rf, positions_rf, fs_imu, plot_trajectory)

    # Get temporal parameters
    step_times_rf = gait_parameter_estimation.get_step_time(hs_lf, hs_rf, fs_imu)
    step_times_lf = gait_parameter_estimation.get_step_time(hs_rf, hs_lf, fs_imu)
    cadence_lf = gait_parameter_estimation.get_cadence(step_times_lf)
    cadence_rf = gait_parameter_estimation.get_cadence(step_times_rf)
    tss_times_lf = gait_parameter_estimation.get_single_support(to_lf, hs_lf, fs_imu)
    tss_times_rf = gait_parameter_estimation.get_single_support(to_rf, hs_rf, fs_imu)

    # Get means and return parameters
    means_data = {
        'Cadence L': [np.mean(cadence_lf)],
        'Single Support L': [np.mean(tss_times_lf)],
        'Step Time L': [np.mean(step_times_lf)],
        'Stride Length L': [np.mean(stride_lengths_lf)],
        'Walking Speed L': [np.mean(walking_speeds_lf)],
        'Cadence R': [np.mean(cadence_rf)],
        'Single Support R': [np.mean(tss_times_rf)],
        'Step Time R': [np.mean(step_times_rf)],
        'Stride Length R': [np.mean(stride_lengths_rf)],
        'Walking Speed R': [np.mean(walking_speeds_rf)]
    }
    means_frame = pd.DataFrame(data=means_data)
    segment_data = SegmentParameters(means_frame, step_times_lf, step_times_rf, stride_lengths_lf, stride_lengths_rf)
    return segment_data

def extract_with_pendulum(acc_data_lumbar, hs_lf, hs_rf, to_lf, to_rf, subject_height):

    # If no events, return None
    if len(hs_lf) < 1 or len(hs_rf) < 1:
        return None

    # Get temporal parameters
    step_times_rf = gait_parameter_estimation.get_step_time(hs_lf, hs_rf, fs_imu)
    step_times_lf = gait_parameter_estimation.get_step_time(hs_rf, hs_lf, fs_imu)
    cadence_lf = gait_parameter_estimation.get_cadence(step_times_lf)
    cadence_rf = gait_parameter_estimation.get_cadence(step_times_rf)
    tss_times_lf = gait_parameter_estimation.get_single_support(to_lf, hs_lf, fs_imu)
    tss_times_rf = gait_parameter_estimation.get_single_support(to_rf, hs_rf, fs_imu)

    # Get spatial parameters if we have subject height
    positions_lumbar = position_estimation.get_positions_lumbar(acc_data_lumbar, hs_lf, hs_rf, fs_imu, plot_positions, None)
    if subject_height:
        step_lengths, walking_speeds = gait_parameter_estimation.get_step_length_speed_lumbar(positions_lumbar, hs_lf, hs_rf, fs_imu, subject_height)
    else:
        step_lengths = []
        walking_speeds = []

    # Assign left/right
    step_lengths_rf = step_lengths[0::2]
    step_lengths_lf = step_lengths[1::2]
    walking_speeds_rf = walking_speeds[0::2]
    walking_speeds_lf = walking_speeds[1::2]
    if hs_lf[0] < hs_rf[0]:
        step_lengths_rf = step_lengths[1::2]
        step_lengths_lf = step_lengths[0::2]
        walking_speeds_rf = walking_speeds[1::2]
        walking_speeds_lf = walking_speeds[0::2]

    # Get means and return parameters
    means_data = {
        'Cadence L': [np.mean(cadence_lf)],
        'Single Support L': [np.mean(tss_times_lf)],
        'Step Length L': [np.mean(step_lengths_lf)],
        'Step Time L': [np.mean(step_times_lf)],
        'Walking Speed L': [np.mean(walking_speeds_lf)],
        'Cadence R': [np.mean(cadence_rf)],
        'Single Support R': [np.mean(tss_times_rf)],
        'Step Length R': [np.mean(step_lengths_rf)],
        'Step Time R': [np.mean(step_times_rf)],
        'Walking Speed R': [np.mean(walking_speeds_rf)]
    }
    means_frame = pd.DataFrame(data=means_data)
    segment_data = SegmentParameters(means_frame, step_times_lf, step_times_rf, [], [])
    return segment_data


def extract_variables(series_in, imu_data, session, subject_height, calibration_data, block_parameters, plot_gait_events=False):
    trial_type = series_in['trial_type']
    imu_onset = float(series_in['adjusted_onset_imu'])
    trial_duration = float(series_in['duration'])
    subject = series_in['subject']
    protocol = series_in['session']

    # Standing audio stroop has no gait parameters
    if trial_type == "Stand_still_and_Aud_Stroop":
        return series_in
    
    # FNP1068 had very strange behavior on gyro data in P1, where left foot cuts off halfway.
    if subject in ['FNP1068'] and protocol in ['protocol1']:
        return series_in

    # Prepare IMU data arrays for condition data
    acc_data_lf, gyro_data_lf, mag_data_lf = extract_condition_data(imu_data, session, "LEFT_FOOT", imu_onset, trial_duration)
    acc_data_rf, gyro_data_rf, mag_data_rf = extract_condition_data(imu_data, session, "RIGHT_FOOT", imu_onset, trial_duration)
    acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar = extract_condition_data(imu_data, session, "LUMBAR", imu_onset, trial_duration)

    # Get turning times
    turn_regions = extract_turn_times(gyro_data_lumbar, mag_data_lumbar, min_turn_limit=1.8, min_region_length=25)

    # We should exclude any turning steps for the straight walking conditions.
    # We should also use different algoritihms for straight walking compared to navigation.
    if trial_type in ["Straight_walking", "Straight_walking_and_Aud_Stroop"]:
        exclude_turns = True
        use_pendulum = False
    else:
        exclude_turns = False
        use_pendulum = True

    # Split up into segments between turns.
    straight_segments = []
    if len(turn_regions) < 1 or not exclude_turns:
        # No turning regions (or consider as one segment): add entire data as a segment
        print("Only using one segment")
        segment = np.arange(0, acc_data_lf.shape[0])
        straight_segments.append(segment)
    else:
        # Add start to first turn and in-between turn regions
        segment_start = 0
        for region in turn_regions:
            region_start = int(region[0])
            # Define segment before turn
            segment_end = region_start-1
            # Add segment to list
            segment = np.arange(segment_start,segment_end)
            straight_segments.append(segment)
            # Start of next segment
            region_end = int(region[-1])
            segment_start = region_end+1
        # Add from last region to end of data, if there is more
        if acc_data_lf.shape[0] > region_end:
            segment = np.arange(segment_start,acc_data_lf.shape[0])
            straight_segments.append(segment)

    # Extract from segment
    all_segment_params = []
    for segment in straight_segments:

        # Skip very short segments
        if len(segment) < 150:
            print("Skipping short segment")
            continue

        segment_start = segment[0]
        segment_end = segment[-1]
        print(f"Handling segment {segment_start}:{segment_end}")
        
        seg_acc_data_lf, seg_gyro_data_lf, seg_mag_data_lf = get_segment_data(acc_data_lf, gyro_data_lf, mag_data_lf, segment_start, segment_end)
        seg_acc_data_rf, seg_gyro_data_rf, seg_mag_data_rf = get_segment_data(acc_data_rf, gyro_data_rf, mag_data_rf, segment_start, segment_end)
        seg_acc_data_lumbar, seg_gyro_data_lumbar, seg_mag_data_lumbar = get_segment_data(acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar, segment_start, segment_end)

        # Get gait events from gyroscope pitch (y axis) data for each foot
        hs_lf, to_lf, ff_lf, stance_lf = gait_event_detection.get_hs_to_ff_gyro_peak(-seg_gyro_data_lf[:,1], fs_imu, plot_figure=plot_gait_events, save_fig_name=None)
        hs_rf, to_rf, ff_rf, stance_rf = gait_event_detection.get_hs_to_ff_gyro_peak(-seg_gyro_data_rf[:,1], fs_imu, plot_figure=plot_gait_events, save_fig_name=None)

        # Skip very short segments with few steps
        if len(hs_lf) < 3 or len(hs_rf) < 3:
            print("Skipping segment with few steps")
            continue

        # Filter signals to smooth them out
        seg_gyro_data_lf, seg_acc_data_lf, seg_mag_data_lf = utility_functions.filter_signal(seg_gyro_data_lf, seg_acc_data_lf, seg_mag_data_lf, fs_imu, plot_filter)
        seg_gyro_data_rf, seg_acc_data_rf, seg_mag_data_rf = utility_functions.filter_signal(seg_gyro_data_rf, seg_acc_data_rf, seg_mag_data_rf, fs_imu, plot_filter)
        seg_gyro_data_lumbar, seg_acc_data_lumbar, seg_mag_data_lumbar = utility_functions.filter_signal(seg_gyro_data_lumbar, seg_acc_data_lumbar, seg_mag_data_lumbar, fs_imu, plot_filter)

        if use_pendulum:
            segment_params = extract_with_pendulum(seg_acc_data_lumbar, hs_lf, hs_rf, to_lf, to_rf, subject_height)
        else:

            # Which direction are we facing?
            segment_mag_avg = np.mean(seg_mag_data_lumbar[:,1])
            direction = 'forward'
            if segment_mag_avg < 0:
                direction = 'backward'

            # Skip very short segments with few strides
            if len(ff_lf) < 3 or len(ff_rf) < 3:
                print("Skipping short segment with few strides")
                continue

            print(f"Handling straight walking segment {segment_start}:{segment_end}, facing {direction}")
            segment_data_lf = [seg_acc_data_lf, seg_gyro_data_lf, seg_mag_data_lf]
            segment_data_rf = [seg_acc_data_rf, seg_gyro_data_rf, seg_mag_data_rf]
            segment_params = extract_straight_walking(segment_data_lf, segment_data_rf, ff_lf, ff_rf, 
                                                      hs_lf, hs_rf, to_lf, to_rf, direction, calibration_data)

        # Append parameters for segment
        all_segment_params.append(segment_params)

    if len(all_segment_params) < 1:
        return series_in
    
    # Create parameters for the whole block and add to passed list
    total_segment_parameters = SegmentParameters.from_multiple_segments(all_segment_params)
    total_segment_parameters.set_trial_type(trial_type)
    block_parameters.append(total_segment_parameters)

    # Add values to series passed into the function and return
    for column in total_segment_parameters.means_data:
        series_in[column] = total_segment_parameters.means_data[column].values[0]
    series_in["Step Count L"] = len(total_segment_parameters.step_times_left)
    series_in["Step Count R"] = len(total_segment_parameters.step_times_right)
    return series_in


if __name__ == "__main__":

    plot_gait_events = 0
    plot_filter = 0
    plot_quaternions_on = 0
    plot_positions = 0
    plot_trajectory = 0

    # Where all the data is stored
    pq_folder = "../Data/imu_data_parquet"
    pq_files = [ f.path for f in os.scandir(pq_folder)]

    # Read calibration data
    calibration_file = "../Data/calibration_stance_data.csv"
    calibration_data = pd.read_csv(calibration_file)

    # Read event data
    event_file = "../Data/all_events_nirs_imu.csv"
    all_events = pd.read_csv(event_file)
    subjects = np.unique(all_events['subject'])
    sessions = ['protocol_1', 'protocol_2', 'protocol_3']

    # Read subject height data
    height_file = "../Data/subject_height_data.csv"
    subject_height_data = pd.read_csv(height_file)
    subject_height_data.rename(columns = {"id_nummer":"subject", "vad_r_din_l_ngd_i_cm":"height_cm"}, inplace = True)

    # Prepare folder for figures
    output_folder_name = os.getcwd() + "/saved_figures"
    if not (os.path.isdir(output_folder_name)):
        os.mkdir(output_folder_name)
    
    # Go through each pq file
    calculated_param_frames = []
    variability_frames = []
    for pq_file in pq_files:

        # Read IMU data
        imu_data = pd.read_parquet(pq_file)
        subject = list(imu_data.columns)[0].split("/")[0]

        # Skip subjects not included in study
        if subject in ['FNP1002','FNP1014']:
            print("Skipping " + subject)
            continue

        # Get subject height for use in pendulum model
        subject_height = subject_height_data[subject_height_data['subject'] == subject]['height_cm'].values[0]
        if np.isnan(subject_height):
            print("WARN: no height data, setting to None")
            subject_height = None

        # Get calibration data
        subject_calibration_data = calibration_data[calibration_data['subject'] == subject]

        print("Using pq file " + pq_file)

        # Go through sessions
        new_events = []
        for session in sessions:
            print("On subject %s, session %s\n" % (subject, session))

            # Get event data and acc data for session
            events = all_events[(all_events["subject"] == subject) & (all_events["session"] == session.replace('_', ''))].copy()
            if events.empty:
                print(f"{subject} does not have {session}, skipping")
                continue

            # FNP1006 seems to have the label RIGHT_WRIST instead of RIGHT_FOOT on protocol 1. Replace.
            if subject == "FNP1006" and session == "protocol_1":
                old_columns = [column for column in imu_data if "RIGHT_WRIST" in column]
                new_columns = [column.replace("WRIST", "FOOT") for column in imu_data if "RIGHT_WRIST" in column]
                col_mapping = dict(zip(old_columns, new_columns))
                imu_data.rename(columns = col_mapping, inplace = True)
            
            # Prepare a list for holding detailed block values
            block_parameters = []

            # Go through each condition
            calculated_params = events.apply(extract_variables, axis=1, imu_data=imu_data, session=session, subject_height=subject_height,
                                             calibration_data=subject_calibration_data, plot_gait_events=plot_gait_events, block_parameters=block_parameters)
            block_numbers = np.arange(start=1, stop=len(calculated_params['onset'])+1)
            calculated_params['block'] = block_numbers
            calculated_param_frames.append(calculated_params)

            # Print results for subject
            print(calculated_params)

            # With data from all blocks, also calculate variability data
            if len(block_parameters) > 1:
                variability_data = calculate_variability_parameters(block_parameters, subject, session.replace('_', ''))
                variability_frames.append(variability_data)
                print(variability_data)

    final_frame_gait = pd.concat(calculated_param_frames).drop(columns=["onset", "adjusted_onset_imu", "duration", "imu_fnirs_lag_seconds", "sample", "value"])
    for col in ['block','trial_type', 'session', 'subject']:
        final_frame_gait.insert(0, col, final_frame_gait.pop(col))
    print(final_frame_gait)
    final_frame_gait.to_csv("../Data/imu_gait_parameters.csv", index=False)

    final_frame_variability = pd.concat(variability_frames)
    print(final_frame_variability)
    final_frame_variability.to_csv("../Data/imu_variability_parameters.csv", index=False)
    print('done')