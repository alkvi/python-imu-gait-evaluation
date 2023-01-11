import os
import pandas as pd
import numpy as np
import utility_functions
import gait_event_detection
import position_estimation
import gait_parameter_estimation
import evaluation_functions

if __name__ == "__main__":

    plot_gait_events = 0
    plot_positions = 0
    plot_filter = 0
    fs_apdm = 128

    # Load target/label data
    apdm_data_file = "Data/apdm_data.csv"
    apdm_data = pd.read_csv(apdm_data_file, encoding="utf-8", header=0)  
    print("Loaded data")
    print("Head")
    print(apdm_data.head())
    print("Tail")
    print(apdm_data.tail())

    # Load xcorr lag
    lag_file = "Data/lag_table.csv"
    lag_table = pd.read_csv(lag_file, encoding="utf-8", header=0)  

    # Load vicon event data
    vicon_event_file = "Data/vicon_event_data.csv"
    vicon_event_table = pd.read_csv(vicon_event_file, encoding="utf-8", header=0)  

    # Load demographic data
    demo_file = "Data/demographic_data.csv"
    demo_table = pd.read_csv(demo_file, encoding="utf-8", header=0)  

    # Prepare folder for figures
    output_folder_name = os.getcwd() + "/saved_figures_m1"
    if not (os.path.isdir(output_folder_name)):
        os.mkdir(output_folder_name)

    # Go through each subject and session
    all_sessions = apdm_data['Session'].unique()
    all_subjects = apdm_data['Subject'].unique()
    all_session_data = []
    for subject in all_subjects:
        for session in all_sessions:

            print("On subject %s, session %s\n" % (subject, session))
            imu_session_data = apdm_data[(apdm_data['Subject'] == subject) & (apdm_data['Session'] == session)]
            lumbar_data = imu_session_data[ imu_session_data['Sensor'] == 'Lumbar']
            acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar, q_data_lumbar = utility_functions.prepare_data(lumbar_data, True)
            
            turning = False
            if "Complex" in session:
                turning = True

            # Read demographic data
            height_data = demo_table[(demo_table['Subject'] == subject)]['Height'].to_numpy()

            # Prepare vicon data for cutting signal
            xcorr_lag = lag_table[(lag_table['Subject'] == subject) & (lag_table['Session'] == session)]['xcorr_lag'].values[0]
            data_cols = [col for col in vicon_event_table.columns if 'Data' in col]
            vicon_event_data = vicon_event_table[(vicon_event_table['Subject'] == subject) & (vicon_event_table['Session'] == session)]
            vicon_event_data = vicon_event_data[data_cols]

            # For determining gait phases, use cut version of IMU signal (to xcorr lag)
            acc_data_lumbar_cut, cut_start_frame = utility_functions.cut_signal_to_events(vicon_event_data, xcorr_lag, acc_data_lumbar)

            # Get gait events from lumbar anterior-posterior (x axis) data
            acc_ap = acc_data_lumbar_cut[:,0]
            save_fig_name = output_folder_name + "/HS TO - " + subject + " - " + session + ".png"
            hs, to, invalid_hs = gait_event_detection.get_hs_to_wavelet(acc_ap, fs_apdm, turning, plot_gait_events, save_fig_name, wavelet_type_straight='gaus1', wavelet_type_turning='gaus1')

            # Assign every other event as left/right
            # Start with right HS, next TO is left TO, so RHS, LTO, LHS, RTO ...
            hs_rf = hs[::2]
            hs_lf = hs[1::2]

            # Remove any TO that's before first RHS
            to = np.delete(to, np.where(to < hs_rf[0])[0])
            to_lf = to[::2]
            to_rf = to[1::2]

            # Re-add cut index
            all_hs = hs + cut_start_frame
            all_to = to + cut_start_frame
            hs_lf = hs_lf + cut_start_frame
            to_lf = to_lf + cut_start_frame
            hs_rf = hs_rf + cut_start_frame
            to_rf = to_rf + cut_start_frame
            if len(invalid_hs) > 0:
                invalid_hs = invalid_hs + cut_start_frame

            # Filter signals to smooth them out
            gyro_data_lumbar, acc_data_lumbar, mag_data_lumbar = utility_functions.filter_signal(gyro_data_lumbar, acc_data_lumbar, mag_data_lumbar, fs_apdm, plot_filter)

            # Get spatial parameters from lumbar sensor
            save_fig_name = output_folder_name + "/Positions - " + subject + " - " + session + ".png"
            positions_lumbar = position_estimation.get_positions_lumbar(acc_data_lumbar, hs_lf, hs_rf, fs_apdm, plot_positions, save_fig_name)
            step_lengths, walking_speeds = gait_parameter_estimation.get_step_length_speed_lumbar(positions_lumbar, hs_lf, hs_rf, fs_apdm, height_data, invalid_hs=invalid_hs)

            # Get temporal parameters
            step_times_rf = gait_parameter_estimation.get_step_time(hs_lf, hs_rf, fs_apdm, invalid_hs=invalid_hs)
            step_times_lf = gait_parameter_estimation.get_step_time(hs_rf, hs_lf, fs_apdm, invalid_hs=invalid_hs)
            cadence_lf = gait_parameter_estimation.get_cadence(step_times_lf)
            cadence_rf = gait_parameter_estimation.get_cadence(step_times_rf)
            tss_times = gait_parameter_estimation.get_single_support(all_to, all_hs, fs_apdm, invalid_hs=invalid_hs)
            tds_times = gait_parameter_estimation.get_double_support(all_hs, all_to, fs_apdm)

            # Assign left/right again
            tds_times_lf = tds_times
            tds_times_rf = tds_times
            step_lengths_rf = step_lengths[0::2]
            step_lengths_lf = step_lengths[1::2]
            walking_speeds_rf = walking_speeds[0::2]
            walking_speeds_lf = walking_speeds[1::2]
            tss_times_rf = tss_times[0::2]
            tss_times_lf = tss_times[1::2]
            if hs_lf[0] < hs_rf[0]:
                step_lengths_rf = step_lengths[1::2]
                step_lengths_lf = step_lengths[0::2]
                walking_speeds_rf = walking_speeds[1::2]
                walking_speeds_lf = walking_speeds[0::2]
                tss_times_rf = tss_times[1::2]
                tss_times_lf = tss_times[0::2]

            # No stride length for lumbar sensor
            stride_lengths_lf = np.zeros(tds_times_lf.shape)
            stride_lengths_rf = np.zeros(tds_times_rf.shape)

            # Only take the means
            means_data_lf = {'Subject': [subject], 'Session': [session], 'Side': ["Left"],
                'Cadence': [np.mean(cadence_lf)],
                'Double Support': [np.mean(tds_times_lf)],
                'Single Support': [np.mean(tss_times_lf)],
                'Step Length': [np.mean(step_lengths_lf)],
                'Step Time': [np.mean(step_times_lf)],
                'Stride Length': [np.mean(stride_lengths_lf)],
                'Walking Speed': [np.mean(walking_speeds_lf)]}
            means_data_rf = {'Subject': [subject], 'Session': [session], 'Side': ["Right"],
                'Cadence': [np.mean(cadence_rf)],
                'Double Support': [np.mean(tds_times_rf)],
                'Single Support': [np.mean(tss_times_rf)],
                'Step Length': [np.mean(step_lengths_rf)],
                'Step Time': [np.mean(step_times_rf)],
                'Stride Length': [np.mean(stride_lengths_rf)],
                'Walking Speed': [np.mean(walking_speeds_rf)]}

            print(means_data_lf)
            print(means_data_rf)
            all_session_data.append(pd.DataFrame(data=means_data_lf))
            all_session_data.append(pd.DataFrame(data=means_data_rf))

    full_frame = pd.concat(all_session_data)

    print("Result")
    print(full_frame)

    save_file_name = "Data/imu_calculated_parameters_wavelet.csv"
    full_frame.to_csv(save_file_name, index=False)

    imu_param_file = save_file_name
    vicon_label_file = "Data/vicon_target_data.csv"
    icc_file_name = "Data/imu_evaluation_wavelet.csv"
    evaluation_functions.generate_boxplot(vicon_label_file, imu_param_file, output_folder_name)
    evaluation_functions.evaluate_accuracy(vicon_label_file, imu_param_file, icc_file_name, output_folder_name, plot_evaluation_figures=False)
    evaluation_functions.generate_icc_figure(icc_file_name, True)

    print("Done")
    print("Saved calculated parameters to %s" % imu_param_file)
    print("Saved calculated ICC to %s" % icc_file_name)