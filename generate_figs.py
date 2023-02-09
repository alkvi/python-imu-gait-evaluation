import os
import utility_functions
import gait_event_detection
import position_estimation
import gait_parameter_estimation
import evaluation_functions

# Generate some figures without running all other calculations.

output_folder_name = os.getcwd() + "/saved_figures_m1"
imu_param_file = "Data/imu_calculated_parameters_wavelet.csv"
vicon_label_file = "Data/vicon_target_data.csv"
icc_file_name = "Data/imu_evaluation_wavelet.csv"
#evaluation_functions.generate_boxplot(vicon_label_file, imu_param_file, output_folder_name)
#evaluation_functions.evaluate_accuracy(vicon_label_file, imu_param_file, icc_file_name, output_folder_name, plot_evaluation_figures=False)
evaluation_functions.generate_icc_figure(icc_file_name, True)

output_folder_name = os.getcwd() + "/saved_figures_m2"
imu_param_file = "Data/imu_calculated_parameters_lumbar.csv"
vicon_label_file = "Data/vicon_target_data.csv"
icc_file_name = "Data/imu_evaluation_lumbar.csv"
#evaluation_functions.generate_boxplot(vicon_label_file, imu_param_file, output_folder_name)
#evaluation_functions.evaluate_accuracy(vicon_label_file, imu_param_file, icc_file_name, output_folder_name, plot_evaluation_figures=False)
evaluation_functions.generate_icc_figure(icc_file_name, True)

output_folder_name = os.getcwd() + "/saved_figures_m3"
imu_param_file = "Data/imu_calculated_parameters_3d.csv"
vicon_label_file = "Data/vicon_target_data.csv"
icc_file_name = "Data/imu_evaluation_3d.csv"
#evaluation_functions.generate_boxplot(vicon_label_file, imu_param_file, output_folder_name)
#evaluation_functions.evaluate_accuracy(vicon_label_file, imu_param_file, icc_file_name, output_folder_name, plot_evaluation_figures=False)
evaluation_functions.generate_icc_figure(icc_file_name, False)