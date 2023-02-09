import os
import re
import pyCompare
import pandas as pd
import pingouin as pg
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import evaluation_functions

def generate_boxplot_mobility_lab(vicon_label_file, imu_param_file, output_folder_name):

    # Load label data
    vicon_label_data = pd.read_csv(vicon_label_file, encoding="utf-8", header=0, usecols=range(1, 18))  

    # Convert session names
    all_session_names = pd.unique(vicon_label_data['Session'])
    for session_name in all_session_names:
        new_session = session_name.replace(" - ", "_")
        new_session = new_session.replace(" ", "")
        new_session = new_session.replace(",", ".")
        vicon_label_data['Session'] = vicon_label_data['Session'].replace(session_name,new_session)
    all_session_names = pd.unique(vicon_label_data['Session'])

    # Load mobility lab parameters
    imu_param_data = pd.read_csv(imu_param_file, encoding="utf-8", header=0)  

    # Go through each subject and session, create a dataframe for comparison of means
    all_data_frames = []
    for index, row in imu_param_data.iterrows():
        subject = row["Subject"]
        session = row["Session"]
        vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]

        vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]
        vicon_mean_cadence = (vicon_session_data['Cadence (R)'].sum() + vicon_session_data['Cadence (L)'].sum()) / (pd.notnull(vicon_session_data['Cadence (R)']).sum() + pd.notnull(vicon_session_data['Cadence (L)']).sum())
        vicon_mean_ds = (vicon_session_data['Double Support (R)'].sum() + vicon_session_data['Double Support (L)'].sum()) / (pd.notnull(vicon_session_data['Double Support (R)']).sum() + pd.notnull(vicon_session_data['Double Support (L)']).sum())
        vicon_mean_ss = (vicon_session_data['Single Support (R)'].sum() + vicon_session_data['Single Support (L)'].sum()) / (pd.notnull(vicon_session_data['Single Support (R)']).sum() + pd.notnull(vicon_session_data['Single Support (L)']).sum())
        vicon_mean_st = (vicon_session_data['Step Time (R)'].sum() + vicon_session_data['Step Time (L)'].sum()) / (pd.notnull(vicon_session_data['Step Time (R)']).sum() + pd.notnull(vicon_session_data['Step Time (L)']).sum())
        vicon_mean_str_l = (vicon_session_data['Stride Length (R)'].sum() + vicon_session_data['Stride Length (L)'].sum()) / (pd.notnull(vicon_session_data['Stride Length (R)']).sum() + pd.notnull(vicon_session_data['Stride Length (L)']).sum())
        vicon_mean_ws = (vicon_session_data['Walking Speed (R)'].sum() + vicon_session_data['Walking Speed (L)'].sum()) / (pd.notnull(vicon_session_data['Walking Speed (R)']).sum() + pd.notnull(vicon_session_data['Walking Speed (L)']).sum())

        imu_mean_cadence = (row['Cadence (R)'] + row['Cadence (L)']) / 2
        imu_mean_ds = ( ((row['Double Support GCT (R)']/100) * row['GCT (R)']) + ((row['Double Support GCT (L)']/100) * row['GCT (L)']) ) / 2
        imu_mean_ss =( ((row['Single Support GCT (R)']/100) * row['GCT (R)']) + ((row['Single Support GCT (L)']/100) * row['GCT (L)']) ) / 2
        imu_mean_stride_length = (row['Step Length (R)'] + row['Step Length (L)']) / 2  # this is actually stride length
        imu_mean_st = (row['Step Time (R)'] + row['Step Time (L)']) / 2
        imu_mean_ws = (row['Walking Speed (R)'] + row['Walking Speed (L)']) / 2

        means_data = {'Subject': [subject], 'Session': [session], 
            'Cadence (vicon)': [vicon_mean_cadence], 'Cadence (imu)': [imu_mean_cadence],
            'Double Support (vicon)': [vicon_mean_ds], 'Double Support (imu)': [imu_mean_ds],
            'Single Support (vicon)': [vicon_mean_ss], 'Single Support (imu)': [imu_mean_ss],
            'Step Time (vicon)': [vicon_mean_st], 'Step Time (imu)': [imu_mean_st],
            'Stride Length (vicon)': [vicon_mean_str_l], 'Stride Length (imu)': [imu_mean_stride_length],
            'Walking Speed (vicon)': [vicon_mean_ws], 'Walking Speed (imu)': [imu_mean_ws],}
        means_data = pd.DataFrame(data=means_data)
        all_data_frames.append(means_data)

    # Create the dataframe from all created frames above
    vicon_imu_frame = pd.concat(all_data_frames)
    print(vicon_imu_frame)

    # Now compare data and get ICCs etc
    measures = ['Cadence', 'Double Support', 'Single Support', 'Step Time', 'Stride Length', 'Walking Speed']

    for measure in measures:

        all_scores = []
        all_vicon_scores = []
        all_imu_scores = []
        all_labels = []

        measure_vicon = measure + " (vicon)"
        measure_wavelet = measure + " (imu)"

        # Straight walking, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_wavelet]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("1.2")
        all_labels.append("1.2")

        # Straight walking, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_wavelet]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.9")
        all_labels.append("0.9")

        # Straight walking, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_wavelet]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.6")
        all_labels.append("0.6")

        # Generate a boxplot
        save_path = output_folder_name + "/Box - " + measure + ".png"
        fig, ax = plt.subplots()
        ax.set_title(measure)
        bplot = ax.boxplot(all_scores, patch_artist=True)
        ax.set_xticklabels(all_labels, rotation="vertical")
        ax.yaxis.grid(True)

        # Separate Vicon/APDM with colors
        colors = ['gold', 'honeydew','gold', 'honeydew','gold', 'honeydew']
        for box in (bplot):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        # Add legend
        viconpatch = mpatches.Patch(facecolor='gold', edgecolor='black', label='Vicon')
        imupatch = mpatches.Patch(facecolor='honeydew', edgecolor='black', label='IMU')
        plt.legend(handles=[viconpatch, imupatch])

        # Add some background color to separate turn/straight
        plt.axvspan(0.5, 6.5, facecolor='lightseagreen', alpha=0.2, zorder=-100)

        y_descr = measure
        if measure in ['Double Support', 'Single Support', 'Step Time']:
            y_descr = "Time [s]"
        elif measure in ["Cadence"]:
            y_descr = measure + " [steps per minute]"
        elif measure in ["Step Length"]:
            y_descr = "Length [m]"
        elif measure in ["Walking Speed"]:
            y_descr = "Speed [m/s]"
        ax.set_ylabel(y_descr)
        ax.set_xlabel("Speed")

        # save plot
        plt.savefig(save_path)

def generate_icc_figure_mobility_lab(icc_file):

    icc_data = pd.read_csv(icc_file, encoding="utf-8", header=0, usecols=range(1, 13))  

    # Drop overall measures
    to_remove  = icc_data['Measure'].str.contains('overall') | icc_data['Measure'].str.contains('all')
    icc_data = icc_data[~to_remove]

    # Remove any small negative values
    icc_data.loc[icc_data['ICC2-1'] < 0, 'ICC2-1'] = 0

    # Re-arrange in order
    cadence_idx = icc_data[icc_data['Measure'].str.contains('Cadence')].index.values
    ds_idx = icc_data[icc_data['Measure'].str.contains('Single Support')].index.values
    ss_idx = icc_data[icc_data['Measure'].str.contains('Double Support')].index.values
    ws_idx = icc_data[icc_data['Measure'].str.contains('Walking Speed')].index.values
    str_idx = icc_data[icc_data['Measure'].str.contains('Stride Length')].index.values
    st_idx = icc_data[icc_data['Measure'].str.contains('Step Time')].index.values
    order = np.concatenate([str_idx, ws_idx, ds_idx, ss_idx, st_idx, cadence_idx])
    icc_data = icc_data.reindex(order)

    # Divide
    straight_data = icc_data['Measure'].str.contains('straight') 
    straight_data = icc_data[straight_data]
    ylabels = straight_data['Measure'].str.split('-').str[0].str.strip() + straight_data['Measure'].str.split('-').str[-1]
    new_labels = []
    for label in ylabels:
        if "0.6" in label:
            new_labels.append(label)
        else:
            new_labels.append(str.split(label, " ")[-1])
    ylabels = np.array(new_labels)

    plt.rcParams.update({'font.size': 15})
    N = ylabels.shape[0]
    y_index = np.arange(N)  # the y locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.barh(y_index, straight_data['ICC2-1'], width, color='lightskyblue')
    ax.set_yticks(y_index + width / 2)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('ICC')
    ax.set_title('ICC(A,1) between Vicon measures and mobility lab')

    # Add regions for ICC scores, Koo and Li (2016): 
    # below 0.50: poor
    # between 0.50 and 0.75: moderate
    # between 0.75 and 0.90: good
    # above 0.90: excellent
    ax.axvline(x=0.9, color='r', linestyle='dashed')
    ax.axvline(x=0.75, color='r', linestyle='dashed')
    ax.axvline(x=0.50, color='r', linestyle='dashed')
    bounding_box=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
    plt.text(0.4+0.05, 2, "poor", size=20, bbox=bounding_box)
    plt.text(0.5+0.09, 2, "moderate", size=20, bbox=bounding_box)
    plt.text(0.75+0.055, 2, "good", size=20, bbox=bounding_box)
    plt.text(0.9+0.03, 2, "excellent", size=20, bbox=bounding_box)


    plt.show()



# Calculates accuracy of mobility lab parameters against vicon data
# and generates boxplots and ICC figure
if __name__ == "__main__":

    # Load target/label data
    vicon_label_file = "Data/vicon_target_data.csv"
    vicon_label_data = pd.read_csv(vicon_label_file, encoding="utf-8", header=0, usecols=range(1, 18))  

    # Convert session names
    all_session_names = pd.unique(vicon_label_data['Session'])
    for session_name in all_session_names:
        new_session = session_name.replace(" - ", "_")
        new_session = new_session.replace(" ", "")
        new_session = new_session.replace(",", ".")
        vicon_label_data['Session'] = vicon_label_data['Session'].replace(session_name,new_session)
    all_session_names = pd.unique(vicon_label_data['Session'])
    print(vicon_label_data)

    # Load mobility lab parameters
    imu_param_file = "Data/apdm_mobility_lab_data.csv"
    imu_param_data = pd.read_csv(imu_param_file, encoding="utf-8", header=0)  
    print(imu_param_data)

    # Prepare folder for figures
    output_folder_name = os.getcwd() + "/saved_figures_mobility_lab"
    if not (os.path.isdir(output_folder_name)):
        os.mkdir(output_folder_name)

    # Go through each subject and session, create a dataframe for comparison of means
    all_data_frames = []
    for index, row in imu_param_data.iterrows():
        subject = row["Subject"]
        session = row["Session"]
        vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]
        means = vicon_session_data.mean()

        vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]
        vicon_mean_cadence = (vicon_session_data['Cadence (R)'].sum() + vicon_session_data['Cadence (L)'].sum()) / (pd.notnull(vicon_session_data['Cadence (R)']).sum() + pd.notnull(vicon_session_data['Cadence (L)']).sum())
        vicon_mean_ds = (vicon_session_data['Double Support (R)'].sum() + vicon_session_data['Double Support (L)'].sum()) / (pd.notnull(vicon_session_data['Double Support (R)']).sum() + pd.notnull(vicon_session_data['Double Support (L)']).sum())
        vicon_mean_ss = (vicon_session_data['Single Support (R)'].sum() + vicon_session_data['Single Support (L)'].sum()) / (pd.notnull(vicon_session_data['Single Support (R)']).sum() + pd.notnull(vicon_session_data['Single Support (L)']).sum())
        vicon_mean_sl = (vicon_session_data['Step Length (R)'].sum() + vicon_session_data['Step Length (L)'].sum()) / (pd.notnull(vicon_session_data['Step Length (R)']).sum() + pd.notnull(vicon_session_data['Step Length (L)']).sum())
        vicon_mean_st = (vicon_session_data['Step Time (R)'].sum() + vicon_session_data['Step Time (L)'].sum()) / (pd.notnull(vicon_session_data['Step Time (R)']).sum() + pd.notnull(vicon_session_data['Step Time (L)']).sum())
        vicon_mean_str_l = (vicon_session_data['Stride Length (R)'].sum() + vicon_session_data['Stride Length (L)'].sum()) / (pd.notnull(vicon_session_data['Stride Length (R)']).sum() + pd.notnull(vicon_session_data['Stride Length (L)']).sum())
        vicon_mean_ws = (vicon_session_data['Walking Speed (R)'].sum() + vicon_session_data['Walking Speed (L)'].sum()) / (pd.notnull(vicon_session_data['Walking Speed (R)']).sum() + pd.notnull(vicon_session_data['Walking Speed (L)']).sum())

        imu_mean_cadence = (row['Cadence (R)'] + row['Cadence (L)']) / 2
        imu_mean_ds = ( ((row['Double Support GCT (R)']/100) * row['GCT (R)']) + ((row['Double Support GCT (L)']/100) * row['GCT (L)']) ) / 2
        imu_mean_ss =( ((row['Single Support GCT (R)']/100) * row['GCT (R)']) + ((row['Single Support GCT (L)']/100) * row['GCT (L)']) ) / 2
        imu_mean_sl = (row['Step Length (R)'] / 2 + row['Step Length (L)'] / 2) / 2  # this is actually stride length, divide by two
        imu_mean_stride_length = (row['Step Length (R)'] + row['Step Length (L)']) / 2  # this is actually stride length
        imu_mean_st = (row['Step Time (R)'] + row['Step Time (L)']) / 2
        imu_mean_ws = (row['Walking Speed (R)'] + row['Walking Speed (L)']) / 2

        means_data = {'Subject': [subject], 'Session': [session], 
            'Cadence (vicon)': [vicon_mean_cadence], 'Cadence (imu)': [imu_mean_cadence],
            'Double Support (vicon)': [vicon_mean_ds], 'Double Support (imu)': [imu_mean_ds],
            'Single Support (vicon)': [vicon_mean_ss], 'Single Support (imu)': [imu_mean_ss],
            'Step Length (vicon)': [vicon_mean_sl], 'Step Length (imu)': [imu_mean_sl],
            'Step Time (vicon)': [vicon_mean_st], 'Step Time (imu)': [imu_mean_st],
            'Stride Length (vicon)': [vicon_mean_str_l], 'Stride Length (imu)': [imu_mean_stride_length],
            'Walking Speed (vicon)': [vicon_mean_ws], 'Walking Speed (imu)': [imu_mean_ws],}
        means_data = pd.DataFrame(data=means_data)
        all_data_frames.append(means_data)

    # Create the dataframe from all created frames above
    vicon_imu_frame = pd.concat(all_data_frames)
    print(vicon_imu_frame)

    # Now compare data and get ICCs etc
    measures = ['Cadence', 'Double Support', 'Single Support', 'Step Length', 'Step Time', 'Stride Length', 'Walking Speed']
    all_scores = []
    for measure in measures:
        measure_vicon = measure + " (vicon)"
        measure_imu = measure + " (imu)"

        # Straight walking, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = evaluation_functions.get_icc_score(score_vicon, score_imu, measure + " - straight - 1.2", output_folder_name, plot_result=False)
        all_scores.append(score_data)

        # Straight walking, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = evaluation_functions.get_icc_score(score_vicon, score_imu, measure + " - straight - 0.9", output_folder_name, plot_result=False)
        all_scores.append(score_data)

        # Straight walking, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = evaluation_functions.get_icc_score(score_vicon, score_imu, measure + " - straight - 0.6", output_folder_name, plot_result=False)
        all_scores.append(score_data)

    print("Result")
    full_frame = pd.concat(all_scores)
    print(full_frame)

    # Save the dataframe to csv
    save_file_name = "Data/mobility_lab_evaluation.csv"
    full_frame.to_csv(save_file_name)

    vicon_label_file = "Data/vicon_target_data.csv"
    icc_file_name = save_file_name
    generate_boxplot_mobility_lab(vicon_label_file, imu_param_file, output_folder_name)
    generate_icc_figure_mobility_lab(icc_file_name)
