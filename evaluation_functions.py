import pyCompare
import pandas as pd
import pingouin as pg
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

def evaluate_accuracy(vicon_label_file, imu_param_file, result_file, output_folder, plot_evaluation_figures=False):

    # Load label/target data
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

    # Load calculated IMU parameters
    imu_param_data = pd.read_csv(imu_param_file, encoding="utf-8", header=0)  
    print(imu_param_data)

    all_sessions = imu_param_data['Session'].unique()
    all_subjects = imu_param_data['Subject'].unique()

    # Go through each subject and session, create a dataframe for comparison of means
    all_data_frames = []
    for subject in all_subjects:
        for session in all_sessions:

            imu_session_data = imu_param_data[(imu_param_data['Subject'] == subject) & (imu_param_data['Session'] == session)]
            imu_means = imu_session_data.mean()
            
            vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]
            vicon_mean_cadence = (vicon_session_data['Cadence (R)'].sum() + vicon_session_data['Cadence (L)'].sum()) / (pd.notnull(vicon_session_data['Cadence (R)']).sum() + pd.notnull(vicon_session_data['Cadence (L)']).sum())
            vicon_mean_ds = (vicon_session_data['Double Support (R)'].sum() + vicon_session_data['Double Support (L)'].sum()) / (pd.notnull(vicon_session_data['Double Support (R)']).sum() + pd.notnull(vicon_session_data['Double Support (L)']).sum())
            vicon_mean_ss = (vicon_session_data['Single Support (R)'].sum() + vicon_session_data['Single Support (L)'].sum()) / (pd.notnull(vicon_session_data['Single Support (R)']).sum() + pd.notnull(vicon_session_data['Single Support (L)']).sum())
            vicon_mean_sl = (vicon_session_data['Step Length (R)'].sum() + vicon_session_data['Step Length (L)'].sum()) / (pd.notnull(vicon_session_data['Step Length (R)']).sum() + pd.notnull(vicon_session_data['Step Length (L)']).sum())
            vicon_mean_st = (vicon_session_data['Step Time (R)'].sum() + vicon_session_data['Step Time (L)'].sum()) / (pd.notnull(vicon_session_data['Step Time (R)']).sum() + pd.notnull(vicon_session_data['Step Time (L)']).sum())
            vicon_mean_str_l = (vicon_session_data['Stride Length (R)'].sum() + vicon_session_data['Stride Length (L)'].sum()) / (pd.notnull(vicon_session_data['Stride Length (R)']).sum() + pd.notnull(vicon_session_data['Stride Length (L)']).sum())
            vicon_mean_ws = (vicon_session_data['Walking Speed (R)'].sum() + vicon_session_data['Walking Speed (L)'].sum()) / (pd.notnull(vicon_session_data['Walking Speed (R)']).sum() + pd.notnull(vicon_session_data['Walking Speed (L)']).sum())

            means_data = {'Subject': [subject], 'Session': [session], 
                'Cadence (vicon)': [vicon_mean_cadence], 'Cadence (IMU)': [imu_means['Cadence']],
                'Double Support (vicon)': [vicon_mean_ds], 'Double Support (IMU)': [imu_means['Double Support']],
                'Single Support (vicon)': [vicon_mean_ss], 'Single Support (IMU)': [imu_means['Single Support']],
                'Step Length (vicon)': [vicon_mean_sl], 'Step Length (IMU)': [imu_means['Step Length']],
                'Step Time (vicon)': [vicon_mean_st], 'Step Time (IMU)': [imu_means['Step Time']],
                'Stride Length (vicon)': [vicon_mean_str_l], 'Stride Length (IMU)': [imu_means['Stride Length']],
                'Walking Speed (vicon)': [vicon_mean_ws], 'Walking Speed (IMU)': [imu_means['Walking Speed']],}
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
        measure_imu = measure + " (IMU)"

        # Straight walking, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - straight - 1.2", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

        # Straight walking, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - straight - 0.9", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

        # Straight walking, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - straight - 0.6", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

        # Turning, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - turn - 1.2", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

        # Turning, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - turn - 0.9", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

        # Turning, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        score_data = get_icc_score(score_vicon, score_imu, measure + " - turn - 0.6", output_folder, plot_result=plot_evaluation_figures)
        all_scores.append(score_data)

    print("Result")
    full_frame = pd.concat(all_scores)
    print(full_frame)

    # Save the dataframe to csv
    full_frame.to_csv(result_file)

def get_icc_score(score_vicon, score_imu, measure_label, output_folder, plot_result):

    # Arrange data in a dataframe as required by the pingouin function
    data_point_indices = np.arange(len(score_vicon))
    data_point_indices = np.tile(data_point_indices,2)
    judge_array_1 = ['Vicon'] * len(score_vicon)
    judge_array_2 = ['APDM (IMU)'] * len(score_imu)
    judge_array = judge_array_1 + judge_array_2
    score_array = np.append(score_vicon, score_imu)
    icc_data_frame = pd.DataFrame({'Data_point': data_point_indices, 'Judge': judge_array, 'Scores': score_array})

    # Get the ICC scores
    icc = pg.intraclass_corr(data=icc_data_frame, targets='Data_point', raters='Judge',
                            ratings='Scores',  nan_policy='omit')
    
    # Extract values of interest
    icc_2_1 = icc[icc['Type'] == "ICC2"]
    icc_val = icc_2_1['ICC'].values[0]
    icc_df1 = icc_2_1['df1'].values[0]
    icc_df2 = icc_2_1['df2'].values[0]
    icc_pval = icc_2_1['pval'].values[0]
    icc_ci = icc_2_1['CI95%'].values[0]
    mean_diff = np.mean(score_imu) - np.mean(score_vicon)
    mean_diff_percent = (np.mean(score_imu) - np.mean(score_vicon)) / np.mean(score_vicon)
    mean_diff_percent = mean_diff_percent * 100

    # Create frame to return
    icc_data = {'Measure': [measure_label], 'Vicon_mean': np.mean(score_vicon), 'Vicon_SD': np.std(score_vicon),
        'APDM_IMU_mean': np.mean(score_imu), 'APDM_IMU_SD': np.std(score_imu), 'Mean_diff': mean_diff, 
        'Mean_diff_%': mean_diff_percent, 'ICC2-1': [icc_val],
        'df1': [icc_df1], 'df2': [icc_df2], 'pval': [icc_pval], 'CI95%': [icc_ci]}
    icc_data = pd.DataFrame(data=icc_data)

    # Also generate a Bland-Altman plot and a score difference plot for these scores.
    if plot_result:
        plot_score_diff(score_vicon, score_imu, measure_label, output_folder)
        plot_bland_altman(score_vicon, score_imu, measure_label, output_folder)
        plt.close()

    return icc_data

def generate_icc_figure(icc_file, is_lumbar):

    icc_data = pd.read_csv(icc_file, encoding="utf-8", header=0, usecols=range(1, 13))  

    # Drop overall measures
    to_remove  = icc_data['Measure'].str.contains('overall') | icc_data['Measure'].str.contains('all')
    icc_data = icc_data[~to_remove]

    # Remove any small negative values
    icc_data.loc[icc_data['ICC2-1'] < 0, 'ICC2-1'] = 0

    # Re-arrange in order
    cadence_idx = icc_data[icc_data['Measure'].str.contains('Cadence')].index.values
    ss_idx = icc_data[icc_data['Measure'].str.contains('Single Support')].index.values
    ds_idx = icc_data[icc_data['Measure'].str.contains('Double Support')].index.values
    sl_idx = icc_data[icc_data['Measure'].str.contains('Step Length')].index.values
    ws_idx = icc_data[icc_data['Measure'].str.contains('Walking Speed')].index.values
    str_l_idx = icc_data[icc_data['Measure'].str.contains('Stride Length')].index.values
    st_idx = icc_data[icc_data['Measure'].str.contains('Step Time')].index.values

    # If data from lumbar sensor, use step length. Otherwise stride length.
    if is_lumbar:
        order = np.concatenate([cadence_idx, st_idx, ds_idx, ss_idx, ws_idx, sl_idx])
    else:
        order = np.concatenate([cadence_idx, st_idx, ds_idx, ss_idx, ws_idx, str_l_idx])
    icc_data = icc_data.reindex(order)

    # Divide
    straight_data = icc_data['Measure'].str.contains('straight') 
    straight_data = icc_data[straight_data]
    turn_data = icc_data['Measure'].str.contains('turn') 
    turn_data = icc_data[turn_data]
    ylabels = straight_data['Measure'].str.split('-').str[0].str.strip() + straight_data['Measure'].str.split('-').str[-1]
    new_labels = []
    for label in ylabels:
        if "1.2" in label:
            new_labels.append(label)
        else:
            new_labels.append(str.split(label, " ")[-1])
    ylabels = np.array(new_labels)

    # Plot
    plt.rcParams["figure.figsize"] = (15,10)
    plt.rcParams.update({'font.size': 20})
    N = ylabels.shape[0]
    y_index = np.arange(N)  # the y locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects_straight = ax.barh(y_index, straight_data['ICC2-1'], width, color='c')
    rects_turn = ax.barh(y_index + width, turn_data['ICC2-1'], width, color='g')
    plt.gca().invert_yaxis() # barh plots in reverse order
    ax.set_yticks(y_index + width / 2)
    ax.set_yticklabels(ylabels)
    ax.legend((rects_straight[0], rects_turn[0]), ('Straight', 'Turn'), loc="lower right")
    ax.set_xlabel('ICC')
    ax.set_title('ICC(A,1) between Vicon measures and IMU')

    # Add regions for ICC scores, Koo and Li (2016): 
    # below 0.50: poor
    # between 0.50 and 0.75: moderate
    # between 0.75 and 0.90: good
    # above 0.90: excellent
    ax.axvline(x=0.9, color='r', linestyle='dashed')
    ax.axvline(x=0.75, color='r', linestyle='dashed')
    ax.axvline(x=0.50, color='r', linestyle='dashed')
    fig.set_tight_layout(True)
    plt.show()

def plot_bland_altman(score_vicon, score_imu, measure_label, output_folder):
    title_str = measure_label
    save_path = output_folder + "/BA - " + measure_label + ".png"
    pyCompare.blandAltman(score_vicon, score_imu, title=title_str, savePath=save_path, percentage=False)

def plot_score_diff(score_vicon, score_imu, measure_label, output_folder):
    # Rows are trials, columns contain mean score for each trial
    combined_frame = pd.concat([score_vicon, score_imu], axis=1)
    trial_names = []
    for i in range(1,combined_frame.shape[0]+1):
        trial_names.append("Trial " + str(i))
    combined_frame['Trial'] = trial_names
    combined_frame['Diff'] = combined_frame.iloc[:,1] - combined_frame.iloc[:,0]

    plt.rcParams["figure.figsize"] = (30,20)
    sns.barplot(data=combined_frame, x="Trial", y="Diff")
    save_path = output_folder + "/score_diff - " + measure_label + ".png"
    plt.savefig(save_path)
    plt.close()

def generate_boxplot(vicon_label_file, imu_param_file, output_folder):

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

    # Load calculated IMU parameters
    imu_param_data = pd.read_csv(imu_param_file, encoding="utf-8", header=0)  

    all_sessions = imu_param_data['Session'].unique()
    all_subjects = imu_param_data['Subject'].unique()

    # Go through each subject and session, create a dataframe for comparison of means
    all_data_frames = []
    for subject in all_subjects:
        for session in all_sessions:

            imu_session_data = imu_param_data[(imu_param_data['Subject'] == subject) & (imu_param_data['Session'] == session)]
            imu_means = imu_session_data.mean()
                
            vicon_session_data = vicon_label_data[(vicon_label_data['Subject'] == subject) & (vicon_label_data['Session'] == session)]
            vicon_mean_cadence = (vicon_session_data['Cadence (R)'].sum() + vicon_session_data['Cadence (L)'].sum()) / (pd.notnull(vicon_session_data['Cadence (R)']).sum() + pd.notnull(vicon_session_data['Cadence (L)']).sum())
            vicon_mean_ds = (vicon_session_data['Double Support (R)'].sum() + vicon_session_data['Double Support (L)'].sum()) / (pd.notnull(vicon_session_data['Double Support (R)']).sum() + pd.notnull(vicon_session_data['Double Support (L)']).sum())
            vicon_mean_ss = (vicon_session_data['Single Support (R)'].sum() + vicon_session_data['Single Support (L)'].sum()) / (pd.notnull(vicon_session_data['Single Support (R)']).sum() + pd.notnull(vicon_session_data['Single Support (L)']).sum())
            vicon_mean_sl = (vicon_session_data['Step Length (R)'].sum() + vicon_session_data['Step Length (L)'].sum()) / (pd.notnull(vicon_session_data['Step Length (R)']).sum() + pd.notnull(vicon_session_data['Step Length (L)']).sum())
            vicon_mean_st = (vicon_session_data['Step Time (R)'].sum() + vicon_session_data['Step Time (L)'].sum()) / (pd.notnull(vicon_session_data['Step Time (R)']).sum() + pd.notnull(vicon_session_data['Step Time (L)']).sum())
            vicon_mean_str_l = (vicon_session_data['Stride Length (R)'].sum() + vicon_session_data['Stride Length (L)'].sum()) / (pd.notnull(vicon_session_data['Stride Length (R)']).sum() + pd.notnull(vicon_session_data['Stride Length (L)']).sum())
            vicon_mean_ws = (vicon_session_data['Walking Speed (R)'].sum() + vicon_session_data['Walking Speed (L)'].sum()) / (pd.notnull(vicon_session_data['Walking Speed (R)']).sum() + pd.notnull(vicon_session_data['Walking Speed (L)']).sum())

            means_data = {'Subject': [subject], 'Session': [session], 
                'Cadence (vicon)': [vicon_mean_cadence], 'Cadence (IMU)': [imu_means['Cadence']],
                'Double Support (vicon)': [vicon_mean_ds], 'Double Support (IMU)': [imu_means['Double Support']],
                'Single Support (vicon)': [vicon_mean_ss], 'Single Support (IMU)': [imu_means['Single Support']],
                'Step Length (vicon)': [vicon_mean_sl], 'Step Length (IMU)': [imu_means['Step Length']],
                'Step Time (vicon)': [vicon_mean_st], 'Step Time (IMU)': [imu_means['Step Time']],
                'Stride Length (vicon)': [vicon_mean_str_l], 'Stride Length (IMU)': [imu_means['Stride Length']],
                'Walking Speed (vicon)': [vicon_mean_ws], 'Walking Speed (IMU)': [imu_means['Walking Speed']],}
            means_data = pd.DataFrame(data=means_data)
            all_data_frames.append(means_data)

    # Create the dataframe from all created frames above
    vicon_imu_frame = pd.concat(all_data_frames)
    print(vicon_imu_frame)

    # Now compare data and get ICCs etc
    measures = ['Cadence', 'Double Support', 'Single Support', 'Step Length', 'Step Time', 'Stride Length', 'Walking Speed']

    for measure in measures:

        all_scores = []
        all_vicon_scores = []
        all_imu_scores = []
        all_labels = []

        measure_vicon = measure + " (vicon)"
        measure_imu = measure + " (IMU)"

        # Straight walking, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("1.2")
        all_labels.append("1.2")

        # Straight walking, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.9")
        all_labels.append("0.9")

        # Straight walking, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Straight0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.6")
        all_labels.append("0.6")

        # Turning, 1.2
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex1.2'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("1.2")
        all_labels.append("1.2")

        # Turning, 0.9
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex0.9'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.9")
        all_labels.append("0.9")

        # Turning, 0.6
        frame = vicon_imu_frame[(vicon_imu_frame['Session'].str.startswith('Complex0.6'))]
        score_vicon = frame[measure_vicon]
        score_imu = frame[measure_imu]
        all_vicon_scores.append(score_vicon)
        all_imu_scores.append(score_imu)
        all_scores.append(score_vicon)
        all_scores.append(score_imu)
        all_labels.append("0.6")
        all_labels.append("0.6")

        # Generate a boxplot
        title_str = measure
        save_path = output_folder + "/Box - " + measure + ".png"
        fig, ax = plt.subplots()
        ax.set_title(measure)
        bplot = ax.boxplot(all_scores, patch_artist=True)
        ax.set_xticklabels(all_labels, rotation="vertical")
        ax.yaxis.grid(True)

        # Separate Vicon/IMU with colors
        colors = ['gold', 'honeydew','gold', 'honeydew','gold', 'honeydew',
                    'gold', 'honeydew','gold', 'honeydew','gold', 'honeydew']
        for box in (bplot):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        # Add legend
        viconpatch = mpatches.Patch(facecolor='gold', edgecolor='black', label='Vicon')
        imupatch = mpatches.Patch(facecolor='honeydew', edgecolor='black', label='IMU')
        plt.legend(handles=[viconpatch, imupatch])

        # Add some background color to separate turn/straight
        plt.axvspan(0.5, 6.5, facecolor='lightseagreen', alpha=0.2, zorder=-100)
        plt.axvspan(6.5, 12.5, facecolor='forestgreen', alpha=0.2, zorder=-100)

        y_descr = measure
        if measure in ['Double Support', 'Single Support', 'Step Time']:
            y_descr = "Time [s]"
        elif measure in ["Cadence"]:
            y_descr = measure + " [steps per minute]"
        elif measure in ["Step Length", "Stride Length"]:
            y_descr = "Length [m]"
        elif measure in ["Walking Speed"]:
            y_descr = "Speed [m/s]"
        ax.set_ylabel(y_descr)

        # show plot
        #plt.show()
        plt.rcParams.update({'font.size': 15})
        fig.set_tight_layout(True)
        plt.savefig(save_path)
        plt.close()