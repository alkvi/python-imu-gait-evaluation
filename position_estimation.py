import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import utility_functions

def get_positions_global(accXYZ, orientation_quaternion, ff, plot_pos, save_fig_name):

    # Scipy library expects quaternion in (x,y,z,w)
    quat_s = np.copy(orientation_quaternion)
    quat_s[:,0] = orientation_quaternion[:,1]
    quat_s[:,1] = orientation_quaternion[:,2]
    quat_s[:,2] = orientation_quaternion[:,3]
    quat_s[:,3] = orientation_quaternion[:,0]
    quat_r = scipy.spatial.transform.Rotation.from_quat(quat_s)

    # Rotate body accelerations to Earth frame (ENU)
    accXYZ_global = quat_r.inv().apply(accXYZ)

    # Remove gravity from measurements (in earth frame)
    gravity_vector = np.transpose(np.array([np.zeros(accXYZ.shape[0]), np.zeros(accXYZ.shape[0]), np.ones(accXYZ.shape[0])*9.81]))
    accXYZ_global = accXYZ_global - gravity_vector

    if plot_pos:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        t = np.arange(accXYZ.shape[0]) 
        ax1.plot(t, accXYZ[:,0], label='X orig')
        ax1.plot(t, accXYZ_global[:,0], label='X rot')
        ax1.set_title('Acc X')
        ax2.plot(t, accXYZ[:,1], label='Y orig')
        ax2.plot(t, accXYZ_global[:,1], label='Y rot')
        ax2.set_title('Acc Y')
        ax3.plot(t, accXYZ[:,2], label='Z orig')
        ax3.plot(t, accXYZ_global[:,2], label='Z rot')
        ax3.set_title('Acc Z')
        plt.legend()
        plt.show()

    # Get an additional adjustment by forcing zero acceleration on FF
    # with function f(x(t)) = t/T*x(t), T = t(ff)
    accXYZ_global_corr = np.copy(accXYZ_global)
    for ff_idx in range(0,len(ff)-1):
        tff = ff[ff_idx]
        next_tff = ff[ff_idx+1]
        for t in range(tff,next_tff):
            accXYZ_global_corr[t, 0] = accXYZ_global[t, 0] * ( (next_tff-t)/(next_tff-tff) )
            accXYZ_global_corr[t, 1] = accXYZ_global[t, 1] * ( (next_tff-t)/(next_tff-tff) )
            accXYZ_global_corr[t, 2] = accXYZ_global[t, 2] * ( (next_tff-t)/(next_tff-tff) )

    if plot_pos:
        zero_line = np.zeros(accXYZ_global.shape)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        t = np.arange(accXYZ.shape[0]) 
        ax1.plot(t, accXYZ_global[:,0], label='X global')
        ax1.plot(t, accXYZ_global_corr[:,0], label='X corr')
        ax1.plot(ff, accXYZ_global_corr[:,0][ff], 'k*', label='FF')
        ax1.plot(t, zero_line,  ':')
        ax1.legend(loc="upper right")
        ax1.set_title('Acc X')
        ax2.plot(t, accXYZ_global[:,1], label='Y global')
        ax2.plot(t, accXYZ_global_corr[:,1], label='Y corr')
        ax2.plot(ff, accXYZ_global_corr[:,1][ff], 'k*', label='FF')
        ax2.plot(t, zero_line,  ':')
        ax2.legend(loc="upper right")
        ax2.set_title('Acc Y')
        ax3.plot(t, accXYZ_global[:,2], label='Z global')
        ax3.plot(t, accXYZ_global_corr[:,2], label='Z corr')
        ax3.plot(ff, accXYZ_global_corr[:,2][ff], 'k*', label='FF')
        ax3.plot(t, zero_line,  ':')
        ax3.legend(loc="upper right")
        ax3.set_title('Acc Z')
        plt.legend()
        plt.show()

    # And replace global Z acc with corrected version
    accXYZ_global[:,2] = accXYZ_global_corr[:,2]

    # Calculate linearly de-drifted velocity between foot flats
    vel = np.zeros(accXYZ_global.shape)
    for ff_idx in range(0,len(ff)-1):
        
        tff = ff[ff_idx]
        next_tff = ff[ff_idx+1]
        if tff == next_tff:
            continue
        
        # Integrate to get velocity
        time_vector = np.arange(tff, next_tff) / 128
        vel[tff:next_tff,0] = scipy.integrate.cumulative_trapezoid(accXYZ_global[tff:next_tff, 0], x=time_vector, axis=0, initial=0)
        vel[tff:next_tff,1] = scipy.integrate.cumulative_trapezoid(accXYZ_global[tff:next_tff, 1], x=time_vector, axis=0, initial=0)
        vel[tff:next_tff,2] = scipy.integrate.cumulative_trapezoid(accXYZ_global[tff:next_tff, 2], x=time_vector, axis=0, initial=0)

        # Get drift between FFs
        vel_interval = vel[tff:next_tff,:]
        vel_interval_t = np.arange(vel_interval.shape[0])
        vel_linear_drift_x = scipy.interpolate.interp1d([vel_interval_t[0], vel_interval_t[-1]], [vel_interval[0,0], vel_interval[-1,0]])
        vel_linear_drift_y = scipy.interpolate.interp1d([vel_interval_t[0], vel_interval_t[-1]], [vel_interval[0,1], vel_interval[-1,1]])
        vel_linear_drift_z = scipy.interpolate.interp1d([vel_interval_t[0], vel_interval_t[-1]], [vel_interval[0,2], vel_interval[-1,2]])
        drift_x = vel_linear_drift_x(vel_interval_t)
        drift_y = vel_linear_drift_y(vel_interval_t)
        drift_z = vel_linear_drift_z(vel_interval_t)
        
        # De-drift
        vel[tff:next_tff,0] = vel[tff:next_tff,0] - drift_x
        vel[tff:next_tff,1] = vel[tff:next_tff,1] - drift_y
        vel[tff:next_tff,2] = vel[tff:next_tff,2] - drift_z

    # Plot velocities of each stride
    if plot_pos:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        for ff_idx in range(0,len(ff)-1):
            tff = ff[ff_idx]
            next_tff = ff[ff_idx+1]
            if tff == next_tff:
                continue
            vel_interval = vel[tff:next_tff,:]
            t = np.arange(vel_interval.shape[0]) 
            ax1.plot(t, vel_interval[:,0])
            ax2.plot(t, vel_interval[:,1])
            ax3.plot(t, vel_interval[:,2])
        ax1.set_title("X velocity profile")
        ax2.set_title("Y velocity profile")
        ax3.set_title("Z velocity profile")
        plt.show()

    # Plot overall velocity
    if plot_pos:
        plt.figure(0)
        t = np.arange(vel.shape[0]) 
        plt.plot(t, vel[:,0], label='x')
        plt.plot(ff, vel[:,0][ff], 'k*', label='FF')
        plt.plot(t, vel[:,1], label='y')
        plt.plot(ff, vel[:,1][ff], 'k*', label='FF')
        plt.plot(t, vel[:,2], label='z')
        plt.plot(ff, vel[:,2][ff], 'k*', label='FF')
        plt.title('Velocity')
        plt.legend()
        plt.show()

    # Integrate velocity to yield position
    pos = np.zeros(vel.shape)
    for ff_idx in range(0,len(ff)-1):
        tff = ff[ff_idx]
        next_tff = ff[ff_idx+1]
        if tff == next_tff:
            continue
        initial_x = pos[tff-1,0]
        initial_y = pos[tff-1,1]
        initial_z = pos[tff-1,2]
        time_vector = np.arange(tff, next_tff) / 128
        pos[tff:next_tff,0] = scipy.integrate.cumulative_trapezoid(vel[tff:next_tff, 0], x=time_vector, axis=0, initial=0) + initial_x
        pos[tff:next_tff,1] = scipy.integrate.cumulative_trapezoid(vel[tff:next_tff, 1], x=time_vector, axis=0, initial=0) + initial_y
        pos[tff:next_tff,2] = scipy.integrate.cumulative_trapezoid(vel[tff:next_tff, 2], x=time_vector, axis=0, initial=0) + initial_z

    # Fix last index
    pos[ff[-1]-1:,0] = pos[ff[-1]-1,0]
    pos[ff[-1]-1:,1] = pos[ff[-1]-1,1]
    pos[ff[-1]-1:,2] = pos[ff[-1]-1,2]

    # Plot position profile of each stride
    if plot_pos:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        for ff_idx in range(0,len(ff)-1):
            tff = ff[ff_idx]
            next_tff = ff[ff_idx+1]
            if tff == next_tff:
                continue
            pos_interval = pos[tff:next_tff,:]
            t = np.arange(pos_interval.shape[0]) 
            ax1.plot(t, pos_interval[:,0])
            ax2.plot(t, pos_interval[:,1])
            ax3.plot(t, pos_interval[:,2])
        ax1.set_title("X position profile")
        ax2.set_title("Y position profile")
        ax3.set_title("Z position profile")
        plt.show()

    # Plot positions
    if plot_pos:
        plt.figure(0)
        t = np.arange(vel.shape[0]) 
        plt.plot(t, pos[:,0], label='x')
        plt.plot(t, pos[:,1], label='y')
        plt.plot(t, pos[:,2], label='z')
        plt.plot(ff, pos[:,0][ff], 'k*', label='FF')
        plt.plot(ff, pos[:,1][ff], 'k*')
        plt.plot(ff, pos[:,2][ff], 'k*')
        plt.title('position')
        plt.legend()

        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()
        
        # Show a 3D trace of positions
        plt.figure(0)
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.plot3D(pos[:,0], pos[:,1], pos[:,2], 'gray')
        ax.plot3D(pos[:,0][ff], pos[:,1][ff], pos[:,2][ff], 'k*')
        ax.set_xlabel('X (East)')
        ax.set_ylabel('Y (North)')
        ax.set_zlabel('Z (Up)')
        utility_functions.set_axes_equal(ax)
        plt.show()

    return pos


def get_positions_lumbar(accXYZ, hs_lf, hs_rf, fs, plot_pos, save_fig_name, stationary_samples=500):

    # Rotate all acc vectors around Y axis (pitch) until mean Z acc is the absolute largest
    acc_z_mean = np.mean(accXYZ[0:stationary_samples,2])
    largest_z = acc_z_mean
    pitch_deg = 0
    pitch_for_largest_z = 0
    for pitch_deg in np.arange(-180,180,0.5):
        global_rot = scipy.spatial.transform.Rotation.from_rotvec(pitch_deg * np.array([0, 1, 0]), degrees=True)
        accXYZ_rot = global_rot.inv().apply(accXYZ)
        acc_z_mean = np.mean(accXYZ_rot[0:stationary_samples,2])

        if acc_z_mean > largest_z:
            largest_z = acc_z_mean
            pitch_for_largest_z = pitch_deg

    print("Largest Z: %f, pitch for largest Z: %f" % (largest_z, pitch_for_largest_z))

    # Rotate body accelerations around Y so largest Z coincides with gravity
    global_rot = scipy.spatial.transform.Rotation.from_rotvec(pitch_for_largest_z * np.array([0, 1, 0]), degrees=True)
    accXYZ_global = global_rot.inv().apply(accXYZ)
    
    # Remove gravity from measurements (in rotated frame)
    grav_offset = largest_z
    gravity_vector = np.transpose(np.array([np.zeros(accXYZ.shape[0]), np.zeros(accXYZ.shape[0]), np.ones(accXYZ.shape[0])*grav_offset]))
    accXYZ_global = accXYZ_global - gravity_vector

    if plot_pos:
        utility_functions.plot_axes_with_hs(accXYZ, accXYZ_global, hs_lf, hs_rf, 'Acc')
        plt.show()

    # Calculate velocity between heel strikes
    all_hs = np.sort(np.concatenate((hs_lf, hs_rf)))
    vel = np.zeros(accXYZ_global.shape)
    for hs_idx in range(0,len(all_hs)-1):
        
        hs = all_hs[hs_idx]
        next_hs = all_hs[hs_idx+1]
        if hs == next_hs:
            continue

        time_vector = np.arange(hs, next_hs) / fs
        vel[hs:next_hs,0] = scipy.integrate.cumulative_trapezoid(accXYZ_global[hs:next_hs, 0], x=time_vector, axis=0, initial=0)
        vel[hs:next_hs,1] = scipy.integrate.cumulative_trapezoid(accXYZ_global[hs:next_hs, 1], x=time_vector, axis=0, initial=0)
        vel[hs:next_hs,2] = scipy.integrate.cumulative_trapezoid(accXYZ_global[hs:next_hs, 2], x=time_vector, axis=0, initial=0)

    if plot_pos:
        plt.figure(0)
        t = np.arange(vel.shape[0]) 
        plt.plot(t, vel[:,0], label='x')
        plt.plot(t, vel[:,1], label='y')
        plt.plot(t, vel[:,2], label='z')
        plt.title('velocity')
        plt.legend()
        plt.show()

    # Integrate velocity between heel strikes to get positions
    pos = np.zeros(vel.shape)
    for hs_idx in range(0,len(all_hs)-1):
        
        hs = all_hs[hs_idx]
        next_hs = all_hs[hs_idx+1]
        if hs == next_hs:
            continue

        initial_x = pos[hs-1,0]
        initial_y = pos[hs-1,1]
        initial_z = pos[hs-1,2]
        time_vector = np.arange(hs, next_hs) / fs
        pos[hs:next_hs,0] = scipy.integrate.cumulative_trapezoid(vel[hs:next_hs, 0], x=time_vector, axis=0, initial=0) + initial_x
        pos[hs:next_hs,1] = scipy.integrate.cumulative_trapezoid(vel[hs:next_hs, 1], x=time_vector, axis=0, initial=0) + initial_y
        pos[hs:next_hs,2] = scipy.integrate.cumulative_trapezoid(vel[hs:next_hs, 2], x=time_vector, axis=0, initial=0) + initial_z

    # Add last index
    pos[all_hs[-1]-1:,0] = pos[all_hs[-1]-1,0]
    pos[all_hs[-1]-1:,1] = pos[all_hs[-1]-1,1]
    pos[all_hs[-1]-1:,2] = pos[all_hs[-1]-1,2]

    # Zijlstra and Hof 2003: To detrend positions, high-pass with 0.1 Hz.
    fn = fs/2
    b, a = scipy.signal.butter(4, 0.1/fn, 'high')
    pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)

    if plot_pos:
        utility_functions.plot_axes_with_hs(pos, pos_filt, hs_lf, hs_rf, 'Pos')

        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()
            
        plt.figure(0)
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.plot3D(pos_filt[:,0], pos_filt[:,1], pos_filt[:,2], 'gray')
        ax.plot3D(pos_filt[:,0][all_hs], pos_filt[:,1][all_hs], pos_filt[:,2][all_hs], 'k*')
        ax.set_xlabel('X (East)')
        ax.set_ylabel('Y (North)')
        ax.set_zlabel('Z (Up)')
        utility_functions.set_axes_equal(ax)
        plt.show()

    return pos_filt