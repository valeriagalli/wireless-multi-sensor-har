'''Generic utility functions for calculations or signal analysis.'''
import math
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt
from scipy import stats


##### CALCULATIONS AND CONVERSIONS #####
def re_im_to_mag_ph(re, im):
	mag = np.sqrt(re**2+im**2)
	ph = np.arctan(im/re)*180/3.14
	return mag, ph


def db_to_nat(val_db):
	return 10**(np.float(val_db)/20)


def calculate_S11_from_Z(mag, ph):
	'''Calculate S11 from impedance magnitude and phase. 
	Returns S11 magnitude (dB) and phase (deg).
	'''
	ph_rad = ph*3.14/180
	Z_complex = mag * (np.cos(ph_rad)+1j*(np.sin(ph_rad)))
	s11 = (Z_complex-50)/(Z_complex+50)
	s11_mag = (abs(s11))
	s11_db = 20*np.log10(s11_mag)
	s11_ph = np.angle(s11)*180/3.14 # equivalent of cmath.phase but for arrays
	return s11_db, s11_ph


def calculate_Z_from_S11(mag, ph):
	'''Calculate Z from S11 magnitude and phase. 
	Returns Z magnitude (dB) and phase (deg).
	'''
	ph_rad = ph*3.14/180
	S11_complex = mag * (np.cos(ph_rad)+1j*(np.sin(ph_rad)))
	Z_complex = 50*(S11_complex+1)/(1-S11_complex)
	Z_mag = (abs(Z_complex))
	Z_ph = np.angle(Z_complex)*180/3.14 # equivalent of cmath.phase but for arrays
	return Z_mag, Z_ph


def calculate_R(ph, X):
	'''Calculate resistance as real part of complex impedance from impedance phase and reactance (imaginary part).'''
	ph_rad = ph*3.14/180
	return X/np.tan(ph_rad)


def calculate_X(ph, R):
	'''Calculate resistance as real part of complex impedance from impedance phase and reactance (imaginary part).'''
	ph_rad = ph*3.14/180
	return R*np.tan(ph_rad)


def calculate_fres_theoretical(L, C):
	return 1/(2*3.14*np.sqrt(np.asarray(L)*np.asarray(C)))


def calculate_strain(x, x0):
	'''Calculate displacement from data of tensile tests (Zwick-Roell).'''
	return (x-x[0])/x0*100


##### DATA PREPARATION AND FILTERING #####
def crop_data(x, y, limit_up):
	return x[y<limit_up], y[y<limit_up]


def bandpass(x, fs, low=0.5, high=10.0, order=4):
	b,a = signal.butter(order, [low/(fs/2), high/(fs/2)], btype='band')
	return signal.filtfilt(b,a,x)


def find_dominant_f(t, x, min_period_sec=0.3, max_period_sec=5.0):
	'''Find dominant frequency of signal x in time t.'''
	fs = 1 / np.mean(np.diff(t))
	# Ensure x is a NumPy array
	x = np.asarray(x).astype(float)
	# FFT
	n = len(x)
	X = np.fft.rfft(x - np.mean(x))
	freqs = np.fft.rfftfreq(n, d=1/fs)
	power = np.abs(X)**2
	dominant_freq_fft = freqs[np.argmax(power)]
	# Power spectral density (Welch)
	f, Pxx = signal.welch(x, fs, nperseg=min(1024, len(x)))
	dominant_freq_welch = f[np.argmax(Pxx)]
	# autocorrelation
	xf = bandpass(x, fs, low=0.5, high=10.0)
	xf = xf - np.mean(xf)
	ac = np.correlate(xf, xf, mode='full')
	ac = ac[len(ac)//2:]   # positive lags
	# ignore zero-lag, find first strong peak after some min lag
	min_lag = int(0.4 * fs)   # ignore very small lags
	peak_lag = np.argmax(ac[min_lag:]) + min_lag
	dominant_freq_peak = fs / peak_lag
	print(f"FFT: {dominant_freq_fft:.2f} Hz\tWelch: {dominant_freq_welch:.2f} Hz"
	   f"\tpeaks: {dominant_freq_fft:.2f} Hz ")
	return np.mean([dominant_freq_fft, dominant_freq_welch, dominant_freq_peak])


def lp_butter(t, x, fc, order=3):
	'''Applies Butteworth low pass filter.
	Input:
		t: time 
		x: raw signal 
		fc: cutoff frequency in Hz
	Returns:
		x1: filtered signal
	'''
	fs = 1/(np.mean(np.diff(t)))
	# print(f'\nSampling frequency = {fs:.2f} Hz')
	nyq = 0.5 * fs		
	error = 1
	while error:
		try:
			b, a, = signal.butter(order, fc/nyq, btype='low', analog=False)
			error = 0
		except ValueError:
			fc -= 1
			print(f'Cut off frequency too high!\nfsampling = {fs},\tcutoff = {fc}')
	x1 = signal.filtfilt(b, a, x)
	return x1


def lp_moving_av(t, x, ws):
	'''Applies running mean as low pass filter.
	Input:
		t: time 
		x: raw signal 
		ws: window size
	Returns:
		x1: filtered signal
	'''
	if not isinstance(x, pd.Series):
		x = pd.Series(x)
	return x.rolling(ws, center=True).mean()


def upsample(x, y, factor):
	'''Resample the data contained in x using a given scaling factor.
	Args:
		x (arr): The array containing the sampling points.
		y (arr): The array containing the data to be interpolated.
		factor (int): The scaling factor for upsampling, e.g. 2 means upsampled data will have 2x elements.
	Returns:
		ynew (arr): The upsampled (interpolated) data.
	'''
	xnew = np.linspace(x[0], x[-1], int(math.floor(len(x)*factor)))
	ynew = np.interp(xnew, x, y)
	return xnew, ynew


def quadratic_fit(f, x):
	'''Fit data with polynomial 2nd order.'''
	model = np.poly1d(np.polyfit(f, x, 2))
	return model, r2_score(x, model(f))


def rolling_mode(labels, win):
	pad = win // 2
	x = np.pad(labels, (pad, pad), mode='edge')
	out = np.empty_like(labels)
	for i in range(len(labels)):
		m = stats.mode(x[i:i+win], keepdims=True).mode[0]
		out[i] = m
	return out


def sync_strain_c(t_strain, strain, t_C, C):
	# Detect peaks
	peaks1, _ = signal.find_peaks(strain, prominence=1) # mm
	peaks2, _ = signal.find_peaks(C, prominence=0.4) # pF
	t_strain_start = t_strain[peaks1[0]]
	t_C_start = t_C[peaks2[0]]
	t_strain_new = t_strain
	time_shift = abs(t_C_start-t_strain_start)
	if t_strain_start>t_C_start:
		# print(f'shifting strain backward by {time_shift:0.2f} seconds') # only for debugging
		t_strain_new -= time_shift
	else:
		t_strain_new += time_shift
		# print(f'shifting strain forward by {time_shift:0.2f} seconds')
	return t_strain_new


def remove_precond_cycles(t_strain, strain, t_C, C, n_precon_cycles):
	# Detect peaks (one per preconditioning cycle)
	peaks1, _ = signal.find_peaks(strain, prominence=1) # mm
	dt = np.mean(np.diff(t_strain[peaks1[0:n_precon_cycles-1]]))
	t_strain_start = t_strain[peaks1[4]] + dt # time of peak of last precondition cycles shifted forward by 1 more second
	strain_new_start_id = np.argmin(np.abs(t_strain-t_strain_start))
	C_new_start_id = np.argmin(np.abs(t_C-t_strain_start))
	t_strain_new = t_strain[strain_new_start_id:]
	strain_new = strain[strain_new_start_id:]
	t_C_new = t_C[C_new_start_id:]
	C_new = C[C_new_start_id:]
	return t_strain_new, strain_new, t_C_new, C_new


##### FIND SIGNAL FEATURES #####
def find_f_res(f, Z, ph, s11_mag_db, exp_n_res):
	'''Identify resonance frequency based on S11 minima, impedance maxima and/or zero 
	crossing of the impedance phase.
	Args:
		f (arr): Frequency points.
		Z (arr): impedance magnitude.
		ph (arr): impedance phase.
		s11_mag_db (arr): S11 magnitude in decibels.
		exp_n_res (int): expected number of resonance (e.g. 2 for 2 sensors).
		
	Returns:
		resonances (arr): Found resonance frequencies.
	'''
	# count s11 local minima
	mins = find_S11_local_mins(s11_mag_db)
	s11_mins = np.around(f[mins][0:exp_n_res], 2)
	# find frequency of Z local maxima
	peaks = find_Z_local_maxs(Z)
	Z_peaks = np.around(f[peaks][0:exp_n_res], 2)
	# find Z phase zero crossing points
	zeros = find_sign_inversion(ph)
	ph_zeros = np.around(f[zeros][0:exp_n_res], 2)
	try:
		resonances = np.concatenate([s11_mins, np.zeros(exp_n_res-s11_mins.shape[0])])
	except ValueError: # ``ValueError: negative dimensions are not allowed`` if more than
					   # 2 peaks are detected, then increase to 3
		resonances = np.concatenate([s11_mins, np.zeros(exp_n_res+1-s11_mins.shape[0])])
	return resonances


def find_S11_local_mins(s11):
	full_span = abs(max(s11)-min(s11))
	mins = signal.find_peaks(-s11, prominence=0.1*full_span)[0] 
	return mins


def find_Z_local_maxs(Z):
	'''Find peaks of impedance magnitude.'''
	full_range = np.max(Z)-np.min(Z)
	maxs = signal.find_peaks(Z, prominence=full_range/2)[0] 
	# maxs = signal.find_peaks(Z, prominence=50)[0] 
	return maxs	


def find_phase_dips(ph):
	'''Find dips (i.e., local minima) of impedance phase.'''
	mins = signal.find_peaks(-ph, prominence=90)[0] # take only local minima of 90 deg or more
	return mins	


def find_sign_inversion(x):
	'''Find change of sign, i.e. 0. It can be used for phase, L or C.
	Args:
		x (arr): Variable of interest whose 0 point is sought.
	Returns:
		i (float): Frequency value of the 0 point.
	'''
	idx = 0
	i = np.nonzero(np.diff(np.sign(x)))
	if i[0].any():
		idx = i[0][0]
	return i[0]


def calculate_deltaC(C):
	'''Calculate and return the capacitance change in each strain cycle.
	Note: the initial value of the capacitance before the strain cycles is used as baseline (instead of each local minimum)
	to avoid any bias due to sensor sagging in the strain release phase.
	'''
	C0 = np.mean(C[0:10])
	dC = (C-C0)/C0*100
	peaks = signal.find_peaks(dC, prominence=2)[0] # 2% change minimum
	C_max = np.mean(C[peaks])
	dC_pc = np.mean(dC[peaks]) 
	dC_abs = dC_pc*C0/100
	return C0, C_max, round(dC_abs, 2), round(dC_pc)


def detect_steps(t, x, tolerance):
	'''Detect staps of x in time (e.g. C from stretching sensors or f if using the whole 
	circuit). This is used for bench tests where sensors are stretched progressively and
	 held for a few seconds. Tolerance is absolute value of expeted fluctuation within
	 the same step.
	'''
	# for convenience also add normalized value dx (subtract baseline)
	baseline = np.mean(x[0:10])
	dx = x-baseline
	# Detect steps: when the value changes beyond small tolerance
	step_changes = np.where(np.abs(np.diff(x)) > tolerance)[0] + 1
	# Add start and end indices
	boundaries = np.concatenate(([0], step_changes, [len(x)]))
	# Compute averages per step
	steps_summary = []
	for i in range(len(boundaries) - 1):
		start_idx, end_idx = boundaries[i], boundaries[i + 1]
		start_time = t[start_idx]
		end_time = t[end_idx - 1]
		avg_x = np.mean(x[start_idx:end_idx])
		avg_dx = np.mean(dx[start_idx:end_idx])
		if end_time-start_time >= 2:
			steps_summary.append({
				'Baseline value': baseline,
				'Step': i + 1,
				'Start Time (s)': start_time,
				'End Time (s)': end_time,
				'Avg absolute value': round(avg_x, 3),
				'Avg delta value': round(avg_dx, 3)
			})
	steps_df = pd.DataFrame(steps_summary)
	return steps_df

