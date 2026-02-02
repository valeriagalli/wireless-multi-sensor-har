'''Utility functions to generate plots.'''
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from brokenaxes import brokenaxes
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import utils.utilities as utilities


##### PLOT AND DISPLAY OPTIONS ##### 
# matplotlib.use('agg') # non GUI backend to avoid matplotlib crash when many plots are generated
plt.style.use('ggplot')
single_col_width = 85 / 25.4      # ≈ 3.35 in 
double_col_width = 180 / 25.4     # ≈ 7.09 in
aspect_ratio = 0.75 # corresponds to 4:3
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']
fontsize=8
plt.rc('font', size=fontsize)         
plt.rc('axes', labelsize=fontsize)   
plt.rc('xtick', labelsize=fontsize, direction='in')   
plt.rc('ytick', labelsize=fontsize, direction='in')    
plt.rc('legend', fontsize=fontsize, loc='best', frameon=False)   
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.pad_inches'] = 0.01
colormap = matplotlib.cm.get_cmap('tab10')
colors = colormap.colors


############### LCR and UTM ###############
def plot_C_LCR(t, C, title):
	'''Plot capacitance from LCR data. 
	Args:
		f (arr): Range of frequency.
		C (arr): Capacitance values.
	Returns:
		fig (figure): Plot.
	'''
	C0 = np.mean(C[0:20]) 
	dC = (C-C0)/C0
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.grid()
	ax[0].plot(t, C)
	ax[0].set_xlabel('Time (s)')
	ax[0].set_ylabel('C (pF)')
	ax[1].plot(t, dC) # zoom in on area with f<f_selfres
	ax[1].set_xlabel('Time (s)')
	ax[1].set_ylabel(r'$\Delta$C/C$_0$')
	plt.grid()	
	plt.tight_layout()
	return fig


def plot_disp_and_C(t_strain, disp, t_C, C, deltaC):
	C0 = np.mean(C[0:20]) 
	dC = (C-C0)/C0
	disp0 = np.mean(disp[0:10]) # initial position at preload (5% strain)
	fig, ax = plt.subplots(3,1, figsize=(single_col_width, single_col_width*aspect_ratio*3), sharex=True)
	ax[0].plot(t_strain, disp-disp0)
	ax[0].set_xlabel('Time (s)')
	ax[0].set_ylabel('Displacement (mm)')
	plt.grid()
	ax[1].plot(t_C, dC) 
	ax[1].set_xlabel('Time (s)')
	ax[1].set_ylabel(r'$\Delta$C/C$_0$')
	plt.grid()
	ax[2].plot(t_C, C) 
	ax[2].set_xlabel('Time (s)')
	ax[2].set_ylabel('C (pF)')
	# ax[2].text(0.05, 0.10, rf'$\Delta C / C_0$ = {deltaC} %', transform=plt.gca().transAxes, fontsize=fontsize,verticalalignment='top',
	# 		bbox=dict(facecolor='white', edgecolor='black', alpha=0.5))
	plt.grid()
	plt.tight_layout()
	return fig


def plot_strain_and_C(t_strain, strain, t_C, C):
	C0 = np.mean(C[0:20]) 
	dC = (C-C0)/C0*100
	plt.close('all')
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2), sharex=True)
	ax[0].plot(t_strain, strain)
	ax[0].set_ylabel('Strain (%)')
	plt.grid()
	ax[1].plot(t_C, dC) 
	ax[1].set_ylabel(r'$\Delta$C/C$_0$ (%)')
	plt.grid()
	fig.supxlabel('Time (s)')
	plt.tight_layout()
	return fig


def plot_strain_and_load(t_strain, strain, load):
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2), sharex=True)
	ax[0].plot(t_strain, strain)
	ax[0].set_ylabel('Strain (%)')
	plt.grid()
	ax[1].plot(t_strain, load) 
	ax[1].set_ylabel('Load (N)')
	plt.grid()
	fig.supxlabel('Time (s)')
	plt.tight_layout()
	return fig


############### Impedance Analyzer (IA) ################## 
def plot_L(f, L, idx, model, r2, title, f_range):
	'''Plot inductance of component across frequency. 
	Args:
		f (arr): Range of frequency.
		L (arr): Inductance values.
		idx (int): Index of self resonance frequency. If self resonance is not in the range, idx is half of the frequency span.
		model (): 2nd order polynomial fit of L for f<f_selfres/2
		r2 (float): Coefficient of determination for the fit.
	Returns:
		fig (figure): Plot.
	'''
	if f_range is not None:
		id_start = np.argmin(abs(f-f_range[0]))
		id_end = np.argmin(abs(f-f_range[1]))
	else:
		id_start = 0
		id_end = -1
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.grid()
	ax[0].plot(f[id_start:id_end], L[id_start:id_end])
	ax[0].set_xlabel('Frequency (MHz)')
	ax[0].set_ylabel(r'L ($\mu H)$')
	f_selfres_str = '$f_{self}$'
	ax[0].text(.01, .99, f'{f_selfres_str} = {f[idx]:.2f} MHz', ha='left', va='top', transform=ax[0].transAxes)
	ax[1].plot(f[0:int(idx/2)], L[0:int(idx/2)]) # zoom in on area with f<f_selfres
	ax[1].plot(f[0:int(idx/2)], model(f[0:int(idx/2)]), '--')
	# ax[1].text(.01, .99, f'{model} \n$R^{2}$={r2:.2f}', ha='left', va='top', transform=ax[1].transAxes)
	ax[1].set_xlabel('Frequency (MHz)')
	ax[1].set_ylabel(r'L ($\mu H)$')
	plt.grid()
	plt.tight_layout()
	return fig


def plot_C(f, C, idx, model, r2, title, f_range):
	'''Plot capacitance of component across frequency. 
	Args:
		f (arr): Range of frequency.
		C (arr): Capacitance values.
		idx (int): Index of self resonance frequency. If self resonance is not in the range, idx is half of the frequency span.
		model (): 2nd order polynomial fit of C for f<f_selfres/2
		r2 (float): Coefficient of determination for the fit.
	Returns:
		fig (figure): Plot.
	'''
	if f_range is not None:
		id_start = np.argmin(abs(f-f_range[0]))
		id_end = np.argmin(abs(f-f_range[1]))
	else:
		id_start = 0
		id_end = -1
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.grid()
	ax[0].plot(f[id_start:id_end], C[id_start:id_end])
	ax[0].set_xlabel('Frequency (MHz)')
	ax[0].set_ylabel('C (pF)')
	f_selfres_str = '$f_{self}$'
	ax[0].scatter(f[idx], C[idx], s=5, color='k', zorder=2)
	ax[0].text(.01, .99, f'{f_selfres_str} = {f[idx]:.2f} MHz', ha='left', va='top', transform=ax[0].transAxes)
	ax[1].plot(f[id_start:int(idx/2)], C[id_start:int(idx/2)]) # zoom in on area with f<f_selfres
	ax[1].plot(f[id_start:int(idx/2)], model(f[id_start:int(idx/2)]), '--')
	# ax[1].text(.01, .99, f'{model} \n$R^{2}$={r2:.2f}', ha='left', va='top', transform=ax[1].transAxes)
	ax[1].set_xlabel('Frequency (MHz)')
	ax[1].set_ylabel('C (pF)')
	plt.grid()
	plt.tight_layout()
	return fig


def plot_C_and_Z(f, Z, ph, R, C, title, f_range):
	if f_range is not None:
		id_start = np.argmin(abs(f-f_range[0]))
		id_end = np.argmin(abs(f-f_range[1]))
	else:
		id_start = 0
		id_end = -1
	fig, ax = plt.subplots(2,2, figsize=(single_col_width*2, single_col_width*aspect_ratio*2))
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.grid()
	ax[0][0].plot(f[id_start:id_end], Z[id_start:id_end], color = colors[0])
	ax[0][0].set_xlabel('Frequency (MHz)')
	ax[0][0].set_ylabel(r'Z (${\Omega}$)')
	ax[0][1].plot(f[id_start:id_end], ph[id_start:id_end], color = colors[1])
	ax[0][1].set_xlabel('Frequency (MHz)')
	ax[0][1].set_ylabel(r'${\theta}$ (deg)')
	ax[1][0].plot(f[id_start:id_end], R[id_start:id_end], color = colors[2])
	ax[1][0].set_xlabel('Frequency (MHz)')
	ax[1][0].set_ylabel(r'R (${\Omega}$)')
	ax[1][1].plot(f[id_start:id_end], C[id_start:id_end], color = colors[3]) # zoom in on area with f<f_selfres
	ax[1][1].set_xlabel('Frequency (MHz)')
	ax[1][1].set_ylabel('C (pF)')
	plt.grid()
	plt.tight_layout()
	return fig


def plot_L_and_Z(f, Z, ph, R, C, title, f_range):
	if f_range is not None:
		id_start = np.argmin(abs(f-f_range[0]))
		id_end = np.argmin(abs(f-f_range[1]))
	else:
		id_start = 0
		id_end = -1
	fig, ax = plt.subplots(2,2, figsize=(single_col_width*2, single_col_width*aspect_ratio*2))
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.grid()
	ax[0][0].plot(f[id_start:id_end], Z[id_start:id_end], color = colors[0])
	ax[0][0].set_xlabel('Frequency (MHz)')
	ax[0][0].set_ylabel(r'Z (${\Omega}$)')
	ax[0][1].plot(f[id_start:id_end], ph[id_start:id_end], color = colors[1])
	ax[0][1].set_xlabel('Frequency (MHz)')
	ax[0][1].set_ylabel(r'${\theta}$ (deg)')
	ax[1][0].plot(f[id_start:id_end], R[id_start:id_end], color = colors[2])
	ax[1][0].set_xlabel('Frequency (MHz)')
	ax[1][0].set_ylabel(r'R (${\Omega}$)')
	ax[1][1].plot(f[id_start:id_end], C[id_start:id_end], color = colors[3]) # zoom in on area with f<f_selfres
	ax[1][1].set_xlabel('Frequency (MHz)')
	ax[1][1].set_ylabel(r'L ($\mu$H)')
	plt.grid()
	plt.tight_layout()
	return fig


###############  Plots from generic instrument ############### 
def plot_Z_mag_and_phase(f, Z, ph):
	'''Plot magnitude and phase and return resonance frequencies.
	Args:
		f (arr): Frequency.
		Z (arr): Impedance magnitude.
		ph (arr): Impedance phase.
	Return
		fig (figure): Plot.
	'''
	fig, ax = plt.subplots(2,1, sharex=True, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	# Z
	# ax[0].set_xlabel('Frequency (MHz)')
	ax[0].set_ylabel('|Z| [\u03A9]')
	ax[0].plot(f, Z, color=colors[1], zorder=1)
	peaks = utilities.find_Z_local_maxs(Z)
	for i in peaks:
		ax[0].scatter(f[i], Z[i], s=5, color='k', zorder=2)
	# phase
	ax[1].set_xlabel('Frequency (MHz)')
	ax[1].set_ylabel('\u03B8 [\u00B0]')
	g = np.zeros(len(f))
	ax[1].plot(f, ph, color= colors[2], zorder=1)
	ax[1].plot(f, g, '--', lw=0.5, color='k')
	i = utilities.find_sign_inversion(ph)
	ax[1].scatter(f[i], ph[i], s=5, color='k', zorder=2)
	return fig


def plot_Z_mag_and_phase_dip(f, Z, ph):
	'''Plot magnitude and phase and return resonance frequencies.
	Args:
		f (arr): Frequency.
		Z (arr): Impedance magnitude.
		ph (arr): Impedance phase.
	Return
		fig (figure): Plot.
	'''
	fig, ax = plt.subplots(2,1, sharex=True, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	# Z
	#ax[0].set_xlabel('Frequency (MHz)')
	ax[0].set_ylabel('|Z| [\u03A9]')
	ax[0].plot(f, Z, color=colors[0], zorder=1)
	peaks = utilities.find_Z_local_maxs(Z)
	for i in peaks:
		ax[0].scatter(f[i], Z[i], s=10, color='k', zorder=2)
		# ax[1].text(max(f)-50, max(Z)-10-10*i, s=f'{f[i]:.2f} MHz') # adjust text position
	# phase 
	ax[1].set_xlabel('Frequency (MHz)')
	ax[1].set_ylabel('\u03B8 [\u00B0]')
	ax[1].plot(f, ph, color= colors[1], zorder=1)
	phase_dips = utilities.find_phase_dips(ph)
	for i in phase_dips:
		ax[1].scatter(f[i], ph[i], s=10, color='k', zorder=2)
	# 	ax[1].text(max(f)-50, max(ph)-10-15*i, s=f'{f[i]:.2f} MHz') # adjust text position
	plt.tight_layout()
	return fig


def plot_S11_mag_and_phase(f, s11, s11_ph):
	'''Plot magnitude and phase and return resonance frequencies.
	Args:
		f (arr): Frequency.
		s11 (arr): Reflection coefficient magnitude.
		s11_ph (arr): Reflection coefficient phase.
	Return
		fig (figure): Plot.
	'''
	fig, ax = plt.subplots(2,1, figsize=(single_col_width, single_col_width*aspect_ratio*2))
	# Z
	ax[0].set_xlabel('Frequency [MHz]')
	ax[0].set_ylabel(r'$S_{11}$ (dB)')
	ax[0].plot(f, s11, color=colors[0])
	throughs = utilities.find_s11_local_mins(s11)
	for i in throughs:
		ax[0][0].scatter(f[i], s11[i], s=5, color='k', zorder=2)
	# phase
	ax[1].set_xlabel('Frequency [MHz]')
	ax[1].set_ylabel(r'$S_{11}$ \u03B8 [\u00B0]')
	ax[1].plot(f, s11_ph, color= colors[1])
	zeros = utilities.find_sign_inversion(s11_ph)
	for i in zeros:
		ax[1].scatter(f[i], s11_ph[i], s=5, color='k', zorder=2)
	return fig


def plot_S11_and_Z(f, z, ph, s11, s11_ph, title, f_range):
	'''Plot magnitude and phase and return resonance frequencies.
	Args:
		f (arr): Frequency.
		z (arr): Impedance magnitude.
		ph (arr): Impedance phase.
		s11 (arr): Reflection coefficient magnitude.
		s11_ph (arr): Reflection coefficient phase.
		f_range (arr): Frequency range to plot (e.g. to zoom in a specific range).
	Return
		fig (figure): Plot.
	'''
	plt.close('all')
	if np.isnan(f).any() or np.isnan(z).any():
		print(f'WARNING: NAN x: {np.isnan(f).any()}, NAN y: {np.isnan(z).any()}')
	fig, ax = plt.subplots(2,2, figsize=(single_col_width*2, single_col_width*aspect_ratio*2))
	if f_range: # if frequency range for the plot has been specified
		fmin = f_range[0]
		fmax = f_range[1]
	else: # otherwise plot the entire frequency range
		fmin = min(f)
		fmax = max(f)
	# S11
	ax[0][0].set_xlim(fmin, fmax)
	ax[0][0].set_xlabel('Frequency (MHz)')
	ax[0][0].set_ylabel(r'$S_{11}$ (dB)')
	ax[0][0].plot(f, s11, color=colors[0], zorder=1)
	s11_min = utilities.find_S11_local_mins(s11)
	for i in s11_min:
		ax[0][0].scatter(f[i], s11[i], s=5, color='k', zorder=2)
	abs_min = min(s11[s11_min]) # find minimum for y scaling
	ax[0][0].set_ylim(abs_min-1, 0)
	# S11 phase
	ax[1][0].set_xlim(fmin, fmax)
	ax[1][0].set_xlabel('Frequency (MHz)')
	ax[1][0].set_ylabel(r'$S_{11}$ $\theta$ $(\degree)$')
	ax[1][0].plot(f, s11_ph, color= colors[1], zorder=1)
	g = np.zeros(len(f))
	ax[1][0].plot(f, g, '--', lw=0.5, color='k')
	zeros = utilities.find_sign_inversion(s11_ph)
	for i in zeros:
		ax[1][0].scatter(f[i], s11_ph[i], s=5, color='k', zorder=2)
	# Z
	ax[0][1].set_xlim(fmin, fmax)
	ax[0][1].set_xlabel('Frequency (MHz)')
	ax[0][1].set_ylabel('|Z| (\u03A9)')
	ax[0][1].plot(f, z, color=colors[2], zorder=1)
	peaks = utilities.find_Z_local_maxs(z)
	for i in peaks:
		ax[0][1].scatter(f[i], z[i], s=5, color='k', zorder=2)
	# phase
	ax[1][1].set_xlim(fmin, fmax)
	ax[1][1].set_xlabel('Frequency (MHz)')
	ax[1][1].set_ylabel(r'$\theta$ $(\degree)$')
	g = np.zeros(len(f))
	ax[1][1].plot(f, ph, color= colors[3], zorder=1)
	ax[1][1].plot(f, g, '--', lw=0.5, color='k')
	zeros = utilities.find_sign_inversion(ph)
	for i in zeros:
		ax[1][1].scatter(f[i], ph[i], s=5, color='k', zorder=2)
	title_polished = title.replace('_', ' ').capitalize()
	fig.suptitle(title_polished)
	plt.tight_layout()
	return fig


###############  Plots related to classification preprocssing (activity tests, VNA data) ############### 
def plot_results_activities(time, f1, f2, lab1, lab2, title, f_res_range):
	'''Plots resonance frequency of each sensor over time for a specific activity.
	Args:
		time (arr): Time stamps
		f1 (arr): Resonance frequency for sensor 1.
		f2 (arr): Resonance frequency for sensor 2.
		lab1 (str): Label for sensor 1 (e.g., ''knee'' or other sensor position).
		lab2 (str): Label for sensor 2 (e.g., ''glute'' or other sensor position).
		title (str): Title for the plot (e.g., which activity).
		f_res_range (arr): Expected resonance frequencies range, to optimize visualization.

	Returns: 
		fig (figure).
	'''
	fig = plt.figure(figsize=(double_col_width, single_col_width*aspect_ratio))
	plt.plot(time, f1, 'o-', ms=0.5, label=lab1)
	plt.plot(time, f2, 'o-', ms=0.5, label=lab2)
	plt.xlabel('Time (s)')
	plt.ylabel('Frequency (MHz)')
	plt.ylim(f_res_range)
	plt.legend()
	title_polished = title.replace('_', ' ').capitalize()
	plt.title(title_polished)
	return fig


def plot_results_activities_delta_f(time, f1, f2, lab1, lab2, title):
	'''Plots resonance frequency of each sensor over time for a specific activity.
	Args:
		time (arr): Time stamps
		f1 (arr): Resonance frequency for sensor 1.
		f2 (arr): Resonance frequency for sensor 2.
		lab1 (str): Label for sensor 1 (e.g., ''knee'' or other sensor position).
		lab2 (str): Label for sensor 2 (e.g., ''glute'' or other sensor position).
		title (str): Title for the plot (e.g., which activity).
		f_res_range (arr): Expected resonance frequencies range, to optimize visualization.

	Returns: 
		fig (figure).
	'''
	fig = plt.figure(figsize=(double_col_width, single_col_width*aspect_ratio))
	plt.plot(time, f1-f1[0], 'o-', ms=0.5, label=lab1)
	plt.plot(time, f2-f2[0], 'o-', ms=0.5, label=lab2)
	plt.xlabel('Time (s)')
	plt.ylabel(r'$\Delta$f (MHz)')
	plt.legend()
	title_polished = title.replace('_', ' ').capitalize()
	plt.title(title_polished)
	return fig


def plot_results_activities_ratio_f(time, f1, f2, lab1, lab2, title):
	'''Plots resonance frequency of each sensor over time for a specific activity.
	Args:
		time (arr): Time stamps
		f1 (arr): Resonance frequency for sensor 1.
		f2 (arr): Resonance frequency for sensor 2.
		lab1 (str): Label for sensor 1 (e.g., ''knee'' or other sensor position).
		lab2 (str): Label for sensor 2 (e.g., ''glute'' or other sensor position).
		title (str): Title for the plot (e.g., which activity).
		f_res_range (arr): Expected resonance frequencies range, to optimize visualization.

	Returns: 
		fig (figure).
	'''
	fig, ax = plt.subplots(figsize=(double_col_width, single_col_width*aspect_ratio))
	ax.plot(time, f1/f1[0], 'o-', ms=0.5, label=lab1)
	ax.plot(time, f2/f2[0], 'o-', ms=0.5, label=lab2)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel(r'$\Delta$f (MHz)')
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.legend()
	title_polished = title.replace('_', ' ').capitalize()
	ax.set_title(title_polished)
	return fig


def plot_activity_labelled(t, f1, labels1, f2, labels2, title):
	'''Plot activity cycles after detection of standing vs activity. 
	Plot full signal (f1, f2) as well to check.'''
	f1_filt = utilities.lp_butter(t, f1, 1)
	f2_filt = utilities.lp_butter(t, f2, 1)
	plt.close()	
	fig = plt.figure(figsize=(single_col_width, single_col_width*aspect_ratio))	
	plt.plot(t, f1_filt, color=colors[3], label='$C_1$')
	plt.plot(t, f2_filt, color=colors[0],  label='$C_2$')
	plt.xlabel('Time (s)')
	plt.ylabel(r'$f_{res}$ (MHz)')
	plt.ylim(20,55)
	plt.legend(loc='center right')
	return fig


def plot_activity_labelled_normalized_f(t, f1, labels1, f2, labels2, title):
	'''Plot activity cycles after detection of standing vs activity. 
	Plot full signal (f1, f2) as well to check.'''
	f1_filt = utilities.lp_butter(t, f1, 4)
	f2_filt = utilities.lp_butter(t, f2, 4)
	f1_norm_filt = f1_filt-np.mean(f1_filt[0:10])
	f2_norm_filt = f2_filt-np.mean(f2_filt[0:10])
	plt.close()	
	fig = plt.figure(figsize=(single_col_width, single_col_width*aspect_ratio))	
	plt.plot(t, f1_norm_filt, color=colors[3],  label='$C_1$')
	plt.plot(t, f2_norm_filt, color=colors[0],   label='$C_2$')
	plt.xlabel('Time (s)')
	plt.ylabel(r'$\Delta$f (MHz)')
	plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.7)
	title_polished = title.replace('_', ' ').capitalize()
	plt.title(title_polished)
	return fig


def plot_activity_merged_labels(t, f1, f2, labels, title):
	'''Plot activity cycles after detection of standing vs activity. 
	Plot full signal (f1, f2) as well to check.'''
	labels_arr = np.array(labels)
	stand_ids = np.where(labels_arr == 0)
	activity_ids = np.where(labels_arr > 0)
	f1_filt = utilities.lp_butter(t, f1, 2)
	f2_filt = utilities.lp_butter(t, f2, 2)
	f1_norm_filt = f1_filt-np.mean(f1_filt[0:10])
	f2_norm_filt = f2_filt-np.mean(f2_filt[0:10])
	plt.close()	
	fig = plt.figure(figsize=(double_col_width, single_col_width*aspect_ratio))	
	plt.plot(t, f1_norm_filt, color=colors[3],  alpha=0.8, label='$C_1$')
	plt.plot(t, f2_norm_filt, color=colors[0],  alpha=0.8,  label='$C_2$')
	plt.plot(t, labels, linewidth=0.8, color='k', label='Label')
	# plt.plot(t[stand_ids], labels[stand_ids], '--', color='k', label='Stand')
	# plt.plot(t[activity_ids], labels[activity_ids], '-^',  color='k', label='Activity')
	plt.xlabel('Time (s)')
	plt.ylabel(r'$\Delta$f (MHz)')
	plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.7)
	title_polished = title.replace('_', ' ').capitalize()
	plt.title(title_polished)
	plt.show()
	return fig


def plot_cycle_starts_ends(X_act, y_act, cycle_start_ids_act, cycle_start_ids, cycle_end_ids_act, cycle_end_ids):
	print(cycle_start_ids_act, cycle_start_ids)
	print(cycle_end_ids_act, cycle_end_ids)
	f1 = X_act['mean1']
	print(f1.index)
	print(f1)
	fig = plt.figure(figsize=(double_col_width, double_col_width*aspect_ratio))	
	plt.plot(np.arange(len(y_act)), y_act, color='green', label='label')
	plt.plot(np.arange(len(y_act)), f1, color='k',)
	# plt.scatter(np.arange(len(y_act))[cycle_start_ids_act], y_act[cycle_start_ids], color='red')
	# plt.scatter(np.arange(len(y_act))[cycle_end_ids_act], y_act[cycle_end_ids], color='blue')
	plt.scatter(np.arange(len(y_act))[cycle_start_ids_act], f1.loc[cycle_start_ids], color='red', label='start points')
	plt.scatter(np.arange(len(y_act))[cycle_end_ids_act], f1.loc[cycle_end_ids], color='blue', label='end points')
	plt.legend(loc='lower right', frameon=True, framealpha=0.7)
	return fig


def plot_removed_standing_and_transition(X, X_fit, mask_fit):
	fig, ax = plt.subplots(2,1)
	ax[0].plot(X[:, 0], color='gray', alpha=0.5, label='f1 mean')
	ax[0].scatter(np.flatnonzero(mask_fit), X_fit[:, 0], 
				color='blue', s=5, label='f1 mean filtered')
	ax[0].set_xlabel('Sample Index')
	ax[0].set_ylabel('Signal Amplitude')
	ax[0].legend()
	ax[1].plot(X[:, 5], color='gray', alpha=0.5, label='f2 mean')
	ax[1].scatter(np.flatnonzero(mask_fit), X_fit[:, 5], 
				color='blue', s=5, label='f2 mean filtered')
	ax[1].set_xlabel('Sample Index')
	ax[1].set_ylabel('Signal Amplitude')
	ax[1].legend()
	plt.suptitle('Original vs Filtered Signal (standing/transition removed)')
	plt.legend()
	# plt.show()
	return fig


def plot_features(df_feat):
	activities = set(df_feat['Activity'])
	print(f'Activities = {activities}')
	n_features = round((df_feat.shape[1]-2)/2)
	print(f'{n_features} features per sensor')
	fig, ax = plt.subplots(2, 1)
	for j, act in enumerate(sorted(activities)):
		df_act = df_feat[df_feat['Activity']==act]
		print(act, set(df_act['y']))
		for i, col in enumerate(df_act.columns[0:n_features]): # sensor 1
			ax[0].plot(df_act[col], color=colors[i], label=col)
		ax[0].plot(df_act['y'], lw= 0.8, color='k')
		ax[0].set_title('Sensor 1')
		for i, col in  enumerate(df_act.columns[n_features:2*n_features]): # sensor 2
			ax[1].plot(df_act[col], color=colors[i], label=col)
		ax[1].plot(df_act['y'], lw=0.8, color='k')
		ax[1].set_title('Sensor 2')
		if j == 0:
			ax[0].legend()
			ax[1].legend()
	plt.show()


###############  Plots related to classification (activity tests, VNA data) ############### 
def plot_learning_curve(model_name, train_sizes, train_mean, train_std, val_mean, val_std):
	fig = plt.figure(figsize=(single_col_width, single_col_width*aspect_ratio))
	plt.plot(train_sizes, train_mean, 'o-', ms=5, label=f'{model_name} Train')
	plt.plot(train_sizes, val_mean, 'o-', ms=5, label=f'{model_name} Validation')
	plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.3)
	plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.3)
	plt.xlabel('Training examples')
	plt.ylabel('F1-weighted')
	plt.title('Learning Curve')
	plt.legend()
	return fig 


def plot_train_val_test_split(X, y, X_train, y_train, X_val, y_val, X_test, y_test):
	fig, ax = plt.subplots(3,1, figsize=(single_col_width, 3*single_col_width*aspect_ratio))
	ax[0].plot(X_train['mean1'], label=r'$C_{1}$')
	ax[0].plot(X_train['mean2'], label=r'$C_{2}$')
	ax[0].plot(y_train, color='k', linewidth=0.8, label='Label')
	ax[0].set_xlabel('Time (s)')
	ax[0].set_ylabel(r'$\Delta$f (MHz)')
	ax[0].set_title('Training data')
	# ax[0].legend(loc='lower right')
	ax[1].plot(X_val['mean1'], label=r'$C_{1}$')
	ax[1].plot(X_val['mean2'], label=r'$C_{2}$')
	ax[1].plot(y_val, color='k', linewidth=0.8, label='Label')
	ax[1].set_title('Validation data')
	ax[1].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.7)
	ax[1].set_xlabel('Time (s)')
	ax[1].set_ylabel(r'$\Delta$f (MHz)')
	ax[2].plot(X_test['mean1'], label=r'$C_{1}$')
	ax[2].plot(X_test['mean2'], label=r'$C_{2}$')
	ax[2].plot(y_test, color='k', linewidth=0.8, label='Label')
	ax[2].set_title('Testing data')
	ax[2].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.7)
	ax[2].set_xlabel('Time (s)')
	ax[2].set_ylabel(r'$\Delta$f (MHz)')
	plt.tight_layout()
	return fig


def plot_confusion_matrix(cm, disp_labels):
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
	disp.plot(cmap='Blues', xticks_rotation=45)
	plt.grid(False) 
	fig_cm = disp.figure_
	return fig_cm


def plot_true_pred(y_true, y_pred, title):
	'''Plot true and predicted labels on the same graph.'''
	plt.close()
	fig = plt.figure() 
	plt.plot(range(0, len(y_true)), y_true, '-s', label='True')
	plt.plot(range(0, len(y_pred)), y_pred, '-o', label='Pred')
	plt.ylabel('Activity')
	title_polished = title.replace('_', ' ').capitalize()
	plt.title(title_polished)
	plt.legend()
	return fig


def plot_clf_reports_multiple_models(clf_report_all):
	'''Plot metrics from the classification report across models.
	Args:
		clr_report_all (DataFrame): DataFrame containing all classification reports from 
									multiple models	.
	Returns:
		fig (figure)
	'''
	# remove the aggregated parameters from classification report and retain only the 
	# metrics for each class (i.e. for each activity)
	df_classes = clf_report_all[~clf_report_all["class"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
	palette = [plt.get_cmap('tab10')(i) for i in range(10)]
	fig, ax = plt.subplots(3, 1, sharex=True,
						figsize=(double_col_width, 3*single_col_width*aspect_ratio))

	# Precision
	sns.barplot(data=df_classes, x="class", y="precision", hue="model", errorbar=None, 
			 ax=ax[0], palette=palette, alpha=0.8)
	ax[0].set_ylim(0, 1.05)
	ax[0].set_ylabel("Precision")
	ax[0].set_xlabel("")
	ax[0].get_legend().remove()

	# Recall
	sns.barplot(data=df_classes, x="class", y="recall", hue="model", errorbar=None, 
			 ax=ax[1], palette=palette, alpha=0.8)
	ax[1].set_ylim(0, 1.05)
	ax[1].set_ylabel("Recall")
	ax[1].set_xlabel("")
	ax[1].get_legend().remove()

	# F1-score
	sns.barplot(data=df_classes, x="class", y="f1_score", hue="model", errorbar=None, 
			 ax=ax[2], palette=palette, alpha=0.8)
	ax[2].set_ylim(0, 1.05)
	ax[2].set_ylabel("F1-score")
	ax[2].set_xlabel("Activity")
	ax[2].get_legend().remove()

	
	# Rotate x-axis labels for readability
	for a in ax:
		a.tick_params(axis='x', rotation=30)

	# Create a single legend outside the plots (center right)
	handles, labels = ax[2].get_legend_handles_labels()
	fig.legend(handles, labels, title="Model",
			loc='center right', bbox_to_anchor=(1.15, 0.5))

	plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for legend on the right
	plt.show()
	return fig
