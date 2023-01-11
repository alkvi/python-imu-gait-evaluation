import numpy as np
import matplotlib.pyplot as plt
import utility_functions

# Step time is time between HS on contralateral legs
def get_step_time(hs_start_side, hs_other_side, fs, invalid_hs=[]):
    step_times =  []
    for hs_idx in range(0,len(hs_start_side)):
        hs = hs_start_side[hs_idx]
        other_side_hs = hs_other_side[hs_other_side > hs]
        if len(other_side_hs) < 1:
            break
        other_side_hs = other_side_hs[0]
        if hs in invalid_hs or other_side_hs in invalid_hs:
            print("Skipping invalid step cycle for step time calculation")
            continue
        hs_diff = other_side_hs - hs
        step_time = hs_diff * 1/fs
        step_times = np.append(step_times, step_time)

    return step_times

# Cadence is amount of steps per minute
# Step times should be in seconds
def get_cadence(step_times):
    cadence = np.divide( np.ones(step_times.shape), step_times ) * 60
    return cadence

# Single support L and R are calculated identically.
# Start with a TO. Get time until next HS.
# This should either be a concatenated vector of HS and TO for both feet, 
# or for one foot individually
def get_single_support(to, hs, fs, invalid_hs=[]):
    tss_times = []
    for to_idx in range(0,len(to)):
        to_frame = to[to_idx]
        next_hs_frame = hs[hs > to_frame]
        if (len(next_hs_frame) == 0):
            break
        next_hs_frame = next_hs_frame[0]
        if next_hs_frame in invalid_hs:
            print("Skipping invalid step cycle for single support calculation")
            continue
        tss = (next_hs_frame - to_frame) * (1/fs)
        tss_times  = np.append(tss_times, tss)
    return tss_times

# Double support consists of both inital and terminal double support.
# Initial is between RHS and LTO. Terminal is between LHS and RTO.
# Add these together to get double support.
# Start with assuming first one is right (would be the same if we started with left)
# This should be a concatenated vector of HS and TO for both feet
def get_double_support(hs, to, fs):
    tds_times = []
    for hs_idx in range(0,len(hs)):
        right_hs_frame = hs[hs_idx]
        left_to_frame = to[to > right_hs_frame]
        if (len(left_to_frame) == 0):
            break
        left_to_frame = left_to_frame[0]
        initial_double_support = (left_to_frame - right_hs_frame) * 1/fs
        left_hs_frame = hs[hs > right_hs_frame]
        if (len(left_hs_frame) == 0):
            break
        left_hs_frame = left_hs_frame[0]
        right_to_frame = to[to > left_hs_frame]
        if (len(right_to_frame) == 0):
            break
        right_to_frame = right_to_frame[0]
        terminal_double_support = (right_to_frame - left_hs_frame) * 1/fs
        tds_times = np.append(tds_times, initial_double_support + terminal_double_support)
    return tds_times

# As above but with separate HS and TO for each foot
def get_double_support_separate(hs_rf, hs_lf, to_rf, to_lf, fs):
    tds_times = []
    for hs_idx in range(0,len(hs_rf)):
        right_hs_frame = hs_rf[hs_idx]
        left_to_frame = to_lf[to_lf > right_hs_frame]
        if (len(left_to_frame) == 0):
            break
        left_to_frame = left_to_frame[0]
        initial_double_support = (left_to_frame - right_hs_frame) * 1/fs
        left_hs_frame = hs_lf[hs_lf > right_hs_frame]
        if (len(left_hs_frame) == 0):
            break
        left_hs_frame = left_hs_frame[0]
        right_to_frame = to_rf[to_rf > left_hs_frame]
        if (len(right_to_frame) == 0):
            break
        right_to_frame = right_to_frame[0]
        terminal_double_support = (right_to_frame - left_hs_frame) * 1/fs
        tds_times = np.append(tds_times, initial_double_support + terminal_double_support)
    return tds_times

# Calculates step length and walking speed for left and right feet, based on positions of lumbar sensor.
# Method based on Ziljstra and Hof 2003 and Del Din et al., 2016 (an inverted pendulum model)
# subject_height must be passed in cm
def get_step_length_speed_lumbar(pos, hs_lf, hs_rf, fs, subject_height, invalid_hs=[]):

    hs_start_side = hs_rf
    hs_other_side = hs_lf
    if hs_lf[0] < hs_rf[0]:
        hs_start_side = hs_lf
        hs_other_side = hs_rf

    # Use Z positions and calculate parameters
    z_positions = pos[:,2]
    step_lengths = []
    walking_speeds = []
    for hs_idx in range(0,len(hs_start_side)-1):

        # Make sure we have enough HS to calculate this cycle
        if hs_idx > len(hs_other_side):
            continue

        # Make sure events are in the correct order
        if hs_other_side[hs_idx] < hs_start_side[hs_idx]:
            print("Skipping wrong order step cycle for step length calculation")
            continue
        
        # Skip cycles involving invalid steps
        if hs_other_side[hs_idx] in invalid_hs or hs_start_side[hs_idx] in invalid_hs:
            print("Skipping invalid step cycle for step length calculation")
            continue

        # The amplitude of changes in vertical position (h)
        # was determined as the difference between highest and
        # lowest position during a step cycle
        # Assuming the lumbar sensor is placed around L5, use factor l = height x 0.53 (Del Din 2016)
        z_interval = z_positions[hs_start_side[hs_idx]:hs_other_side[hs_idx]]
        delta_z = z_interval.max() - z_interval.min()
        delta_z = abs(delta_z)
        step_length = 2*np.sqrt(2*(subject_height/100)*0.53*delta_z - np.power(delta_z,2))
        step_lengths = np.append(step_lengths, step_length)
        hs_diff = hs_other_side[hs_idx] - hs_start_side[hs_idx]
        walking_speed = step_length / ( hs_diff / fs )
        walking_speeds = np.append(walking_speeds, walking_speed)

    return step_lengths, walking_speeds

# Stride lengths and walking speeds from 3D positions of foot sensor.
# Stride: how much one foot travels from FF to FF.
def get_stride_length_walking_speed_foot(ff_times, positions, fs_apdm, plot_traj):
    
    # Set parameters
    heading_steps = 2
    min_lim = 0
    max_lim = 2

    # Get start and end positions
    end_ff = ff_times[-1]
    start_pos = positions[0,:]
    end_pos = positions[end_ff,:]

    # Set up a plot for showing trajectories
    if plot_traj:
        plt.figure(0)
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
        ax.plot3D(start_pos[0], start_pos[1], start_pos[2], 'g*')
        ax.plot3D(end_pos[0], end_pos[1], end_pos[2], 'r*')
        ax.set_xlabel('X (East)')
        ax.set_ylabel('Y (North)')
        ax.set_zlabel('Z (Up)')
    
    # Calculate strides between foot flats
    stride_lengths =  []
    walking_speeds =  []
    for ff_idx in range(0,len(ff_times)-1):
        ff = ff_times[ff_idx]
        next_ff = ff_times[ff_idx+1]
        pos_first = positions[ff,:]
        pos_second = positions[next_ff,:]
        pos_diff = pos_second - pos_first

        # Project stride vector into a local heading direction.
        # If heading_steps additional FFs exist, use the position of the last FF.
        # Otherwise, use the final position.
        if ff_idx+heading_steps < len(ff_times):
            heading_ff = ff_times[ff_idx+heading_steps]
            heading_vector = positions[heading_ff,:] - positions[ff,:]
        else:
            heading_vector = end_pos - positions[ff,:]

        # Perform scalar projection
        stride_length = np.dot(pos_diff, heading_vector) / np.linalg.norm(heading_vector)

        # Make sure stride is within certain limits
        if stride_length < min_lim or stride_length > max_lim:
            print("Skipping stride length with value %f (outside limit)" % (stride_length))
        else:
            stride_lengths = np.append(stride_lengths, stride_length)
            walking_speed = stride_length / ( (next_ff-ff) / fs_apdm )
            walking_speeds = np.append(walking_speeds, walking_speed)

        # Plot a trajectory for each individual stride
        if plot_traj:
            stride_line_x = [pos_first[0], pos_second[0]]
            stride_line_y = [pos_first[1], pos_second[1]]
            stride_line_z = [pos_first[2], pos_second[2]]
            heading_unit_vector = heading_vector / np.linalg.norm(heading_vector)
            stride_vector_proj = heading_unit_vector * stride_length
            stride_line_proj_x = [pos_first[0], pos_first[0] + stride_vector_proj[0]]
            stride_line_proj_y = [pos_first[1], pos_first[1] + stride_vector_proj[1]]
            stride_line_proj_z = [pos_first[2], pos_first[2] + stride_vector_proj[2]]
            if ff_idx not in [0]:
                ax.plot3D(positions[ff,0], positions[ff,1], positions[ff,2], 'k*')
            if ff_idx not in [len(ff_times)-2]:
                ax.plot3D(positions[next_ff,0], positions[next_ff,1], positions[next_ff,2], 'k*')
            ax.plot3D(stride_line_proj_x, stride_line_proj_y, stride_line_proj_z, 'cyan')
            ax.plot3D(stride_line_x, stride_line_y, stride_line_z, 'blue')

    # Adjust and show trajectory
    if plot_traj:
        utility_functions.set_axes_equal(ax)
        plt.show()

    return stride_lengths, walking_speeds
