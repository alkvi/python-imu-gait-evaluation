import os 
import pywt
import scipy
import numpy as np
from matplotlib import pyplot as plt

# Get scale for a wavelet and signal frequency
def get_scales(wavelet, max_freq, fs):
    fc = pywt.central_frequency(wavelet)
    fa = max_freq
    sampling_period = 1/fs
    scales = fc/(fa*sampling_period)
    return scales

# Get a wavelet object
# Note: pywt has been modified to allow cwt for discrete wavelets (including db2), to replicate MATLAB cwt behavior
def get_wavelet(wavelet_name):
    discrete_wavelets = ['db2']
    continuous_wavelets = ['gaus1', 'gaus2']
    if wavelet_name in discrete_wavelets:
        wavelet = pywt.Wavelet(wavelet_name)
    elif wavelet_name in continuous_wavelets:
        wavelet = pywt.ContinuousWavelet(wavelet_name)
    else:
        print("ERROR: wavelet type %s not a valid type for this method" % wavelet_name)
        raise ValueError
    return wavelet

# Calculates HS and TO with CWT, based on Pham et al., 2017.
def get_hs_to_wavelet(acc_ap, fs, turning, plot_figure, save_fig_name, wavelet_type_straight="gaus1", wavelet_type_turning="gaus1"):

    if (turning):
        print('This is a turning dataset, will use %s' % wavelet_type_turning)
    else:
        print('This is a straight walking dataset, will %s' % wavelet_type_straight)

    # Preprocess acceleration data (see paragraph "Extraction of HS and TO from
    # IMU" lines 1-4 in Pham et al., 2017)
    acc_detrend = scipy.signal.detrend(acc_ap)
    fn = fs/2
    b, a = scipy.signal.butter(2, 10/fn, 'low')
    acc_ap_pp = scipy.signal.filtfilt(b, a, acc_detrend, axis=0)

    # Integrate detrended and filtered acceleration data (see paragraph
    # "Extraction of HS and TO from IMU" lines 4-7 in Pham et al., 2017)
    signal_len = acc_ap_pp.shape[0]
    time_vector = np.arange(0, signal_len) / fs
    acc_int = scipy.integrate.cumulative_trapezoid(acc_ap_pp, x=time_vector, axis=0, initial=0)

    # Find the dominant frequency of the acceleration (see paragraph
    # "Extraction of HS and TO from IMU" lines 8-15 in Pham et al., 2017)
    half_idx = int(signal_len/2)
    f_x = scipy.fft.fft(acc_ap_pp) / signal_len
    freqs = np.fft.fftfreq(signal_len, d=1/fs)
    freqs_halfside = freqs[0:half_idx]
    power = 10*np.log10(np.abs(f_x[0:half_idx]))
    max_power = power.max()
    max_idx = power.argmax()
    max_freq = freqs_halfside[max_idx]
    print("Dominant frequency: %f, with power: %f" % (max_freq, max_power))

    # Select wavelet for smoothing (see paragraph "Extraction of HS and TO from IMU" lines 4-7 in Pham et al., 2017)
    wavelet_dcwt1 = pywt.ContinuousWavelet('gaus1')
    scales = get_scales(wavelet_dcwt1, max_freq, fs)

    # Differentiate integrated signal to smooth acceleration signal
    acc_wave, _ = pywt.cwt(acc_int, scales, wavelet_dcwt1)
    acc_wave = -acc_wave[0,:] # Invert to match original signal, see Pham and McCamley
    acc_wave = np.real(acc_wave)

    # Find heel strike events (see Fig. 2 in Pham et al., 2017)
    # Here, we want to find local minima. Apply findpeaks to negative signal.
    acc_wave_detrended = scipy.signal.detrend(acc_wave)
    hs_idx, _ = scipy.signal.find_peaks(-acc_wave_detrended)

    # Find toe off events (see paragraph "Adaptation of the cwt to the Steps
    # in the Home-Like Assessment" in Pham et al., 2017).
    # Paper recommends db2 for straight walking and gaus2 for turning, 
    # although here gaus1 was found to work best.
    if (turning):
        wavelet_dcwt2 = get_wavelet(wavelet_type_turning)
    else:
        wavelet_dcwt2 = get_wavelet(wavelet_type_straight)
    scales = get_scales(wavelet_dcwt2, max_freq, fs)
    acc_wave_2, _ = pywt.cwt(acc_wave, scales, wavelet_dcwt2)
    acc_wave_2 = -acc_wave_2[0,:]
    acc_wave_2 = np.real(acc_wave_2)

    # Find maxima of differentiated signal to get TO
    to_idx, _ = scipy.signal.find_peaks(acc_wave_2)
    
    # Peak selection by magnitude. From paper:
    # "HS/TO was as follows: magnitude >40% of the mean of all peaks detected by
    # the findpeaks function."
    # The limit has here been modified to 0.2 for HS which was found to work better.
    pks_hs = acc_wave[hs_idx]
    pks_to = acc_wave_2[to_idx]
    hs_mean = np.mean(pks_hs)
    to_mean = np.mean(pks_to)
    hs_selected_idx = hs_idx[np.where(pks_hs < hs_mean*0.2)]
    to_selected_idx = to_idx[np.where(pks_to > to_mean*0.4)]

    # Some validation.
    # HS has to be followed by TO. otherwise mark as invalid.
    invalid_hs_idx = []
    for i in range(0, len(hs_selected_idx)-1):
        hs = hs_selected_idx[i]
        next_hs = hs_selected_idx[i+1]
        next_to = to_selected_idx[to_selected_idx > hs]
        if len(next_to) < 1:
            # No found TO. Mark HS as invalid.
            invalid_hs_idx = np.append(invalid_hs_idx, hs)
        else:
            # Found next TO. Make sure it's in the right order compared to upcoming HS.
            next_to = next_to[0]
            if next_hs < next_to:
                invalid_hs_idx = np.append(invalid_hs_idx, hs)
        
    # If the very last HS is not followed by TO, mark as invalid.
    if hs_selected_idx[-1] > to_selected_idx[-1]:
        invalid_hs_idx = np.append(invalid_hs_idx, hs_selected_idx[-1])

    # Convert to int array
    invalid_hs_idx = np.array(invalid_hs_idx)
    invalid_hs_idx = invalid_hs_idx.astype(int)

    # Get the corresponding times
    hs_times = time_vector[hs_selected_idx]
    to_times = time_vector[to_selected_idx]
    invalid_hs_times = time_vector[invalid_hs_idx]

    if plot_figure:
        print("Plotting")
        plt.figure(0)
        plt.plot(time_vector, acc_ap_pp, 'b--', label='acc_ap')
        plt.plot(time_vector, acc_wave_detrended, color="grey", linestyle="dotted", label='dcwt1')
        plt.plot(time_vector, acc_wave_2, 'k', label='dcwt2')
        plt.plot(hs_times, acc_wave_detrended[hs_selected_idx], linestyle='None', color='red', label='HS', marker="*",  markersize=10)
        plt.plot(to_times, acc_wave_2[to_selected_idx], linestyle='None', color='lime', label='TO', marker="*",  markersize=10)
        plt.plot(invalid_hs_times, acc_wave_detrended[invalid_hs_idx], linestyle='None', color='orange', label='HS invalid', marker="*",  markersize=10)
        plt.title(save_fig_name)
        plt.xlabel('Time [s]')
        plt.legend(loc="upper right")
        
        if save_fig_name is not None:
            figure = plt.gcf() # get current figure
            plt.rcParams["figure.figsize"] = (16,8)
            plt.rcParams.update({'font.size': 20})
            plt.savefig(save_fig_name, dpi = 100)
            plt.close()
        else:
            plt.show()

    return hs_selected_idx, to_selected_idx, invalid_hs_idx


# Calculates HS, TO and FF with gyroscope data, based on Salarian et al., 2004.
def get_hs_to_ff_gyro_peak(gyro_data, fs, plot_figure, save_fig_name):

    # Normalize data 
    gyro_data_norm = scipy.stats.zscore(gyro_data)

    # Prepare a filter
    # Order 48 FIR, low-pass, cutoff 30 Hz
    Fn = fs/2
    b = scipy.signal.firwin(48, 30/Fn, pass_zero="lowpass")

    # First identify maxima of signal to identifty mid-swing
    # Salarian et al 2004: Those peaks that were larger than 50 deg/s were candidates
    # If multiple adjacent peaks within a maximum distance of 500 ms were detected,
    # the peak with the highest amplitude was selected and the others were discarded
    min_midswing_height = 1 # we have a normalized signal, not an absolute value
    min_peak_distance_time = 0.5
    min_midswing_distance_samples = int(fs*min_peak_distance_time)
    ms_idx, _ = scipy.signal.find_peaks(gyro_data_norm, height=min_midswing_height, distance=min_midswing_distance_samples)
    
    # Salarian et al 2004: local minimum peaks of shank signal inside interval -1.5s +1.5s were searched. 
    # The nearest local minimum after MS was selected as IC.
    search_interval = 1.5
    search_interval_samples = int(fs*search_interval)

    all_hs = []
    all_to = []
    for i in range(0,len(ms_idx)):
        t_ms_idx = ms_idx[i]
        start_idx = int(t_ms_idx - search_interval_samples)
        end_idx = int(t_ms_idx + search_interval_samples)
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(gyro_data_norm)-1:
            end_idx = len(gyro_data_norm)-1

        signal_interval = gyro_data_norm[start_idx:end_idx]
            
        # Salarian et al 2004: to smooth the signal and to get rid of spurious peaks, 
        # the signal was filtered using a low-pass FIR filter with cutoff frequency 
        # of 30 Hz and pass-band attenuation of less than 0.5 dB.
        signal_interval_filt = scipy.signal.filtfilt(b, [1.0], signal_interval)

        # The nearest local minimum after the t_ms was selected as IC (i.e. HS).
        min_peak_height = 0.1
        signal_interval_filt_hs = signal_interval_filt[search_interval_samples:-1]
        
        # In case we have a very tiny segment, use argmax for peak. Otherwise find_peaks.
        if (len(signal_interval_filt_hs)) < 3:
            hs_idx = np.argmax(signal_interval_filt_hs)
        else:
            hs_idx, properties = scipy.signal.find_peaks(-signal_interval_filt_hs, height=min_peak_height)
        
        # Add HS if found
        if len(hs_idx) >= 1:
            hs_idx = hs_idx[0]
            all_hs = np.append(all_hs, t_ms_idx+hs_idx)
        
        # Salarian et al 2004: the minimum prior to t_ms with amplitude less than -20 deg/s was
        # selected as the terminal contact (i.e. TO).
        if search_interval_samples < t_ms_idx:
            signal_interval_filt_to = signal_interval_filt[0:search_interval_samples]
        else:
            signal_interval_filt_to = signal_interval_filt[0:t_ms_idx]

        # Find TO and add if found
        min_peak_height = 1
        min_to_distance = 0.15
        min_to_distance_samples = int(fs*min_to_distance)
        to_idx, properties = scipy.signal.find_peaks(-signal_interval_filt_to, height=min_peak_height, distance=min_to_distance_samples)
        if len(to_idx) >= 1:
            to_idx = to_idx[-1]
            all_to = np.append(all_to, start_idx+to_idx)

    # Make sure we have integers
    all_hs = all_hs.astype(int)
    all_to = all_to.astype(int)
    all_ms = ms_idx.astype(int)
    
    # Also return foot flat and stance time points.
    # Stance is time between HS and TO on same leg.
    # Identify foot flat times as when angular velocity absolute value
    # is below a certain threshold, during each stance phase.
    all_tff = []
    all_stance = []
    for i in range(0,len(all_hs)):
        
        hs = all_hs[i]
        next_to = all_to[all_to > hs]
        if (len(next_to) < 1):
            continue
        next_to = next_to[0]
        
        gyro_interval_times = np.array(range(hs,next_to))
        gyro_interval = gyro_data_norm[hs:next_to]
        gyro_interval_flat_time = gyro_interval_times[np.where(abs(gyro_interval) < 0.2)[0]]
        
        # Only take instants
        if len(gyro_interval_flat_time) < 1:
            continue
        median_idx = int(np.floor(len(gyro_interval_flat_time)/2))
        ff_median = gyro_interval_flat_time[median_idx]
        all_tff = np.append(all_tff, ff_median)
        all_stance = np.append(all_stance, gyro_interval_flat_time)

    all_tff = all_tff.astype(int)
    all_stance = all_stance.astype(int)

    if plot_figure:
        plt.figure()
        x = np.arange(len(gyro_data_norm))
        time_vector = np.arange(0, len(gyro_data_norm)) / fs
        plt.plot(time_vector, gyro_data_norm, 'b', label='Gyro')
        plt.plot(time_vector[all_hs], gyro_data_norm[all_hs], linestyle='None', color='red', label='HS', marker="*",  markersize=10)
        plt.plot(time_vector[all_to], gyro_data_norm[all_to], linestyle='None', color='lime', label='TO', marker="*",  markersize=10)
        plt.plot(time_vector[all_tff], gyro_data_norm[all_tff], linestyle='None', color='black', label='FF', marker="*",  markersize=10)
        plt.plot(time_vector[all_ms], gyro_data_norm[all_ms], linestyle='None', color='yellow', label='MS', marker="*",  markersize=10)
        plt.title(save_fig_name)
        plt.xlabel('Time [s]')
        plt.legend(loc="upper right")
        
        if save_fig_name is not None:
            figure = plt.gcf() # get current figure
            plt.rcParams["figure.figsize"] = (16,8)
            plt.rcParams.update({'font.size': 20})
            plt.savefig(save_fig_name, dpi = 100)
            plt.close()
        else:
            plt.show()

    return all_hs, all_to, all_tff, all_stance