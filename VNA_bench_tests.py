'''Plot data from VNA files. This module is used as first analysis on the raw data 
from the Keyisght VNA (either csv files or s1p files).

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	
	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python main_VNA.py

'''
import pandas as pd
from pathlib import Path, PurePath
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.read as read
import utils.interactive as interactive
from matplotlib import pyplot as plt


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
	mins = utilities.find_S11_local_mins(s11_mag_db)
	s11_mins = np.around(f[mins][0:exp_n_res+1], 2)
	# find frequency of Z local maxima
	peaks = utilities.find_Z_local_maxs(Z)
	Z_peaks = np.around(f[peaks][0:exp_n_res+1], 2)
	# find Z phase zero crossing points
	zeros = utilities.find_sign_inversion(ph)
	ph_zeros = np.around(f[zeros][0:exp_n_res+1], 2)
	try:
		resonances = np.concatenate([s11_mins, np.zeros(exp_n_res-s11_mins.shape[0])])
	except ValueError: # ``ValueError: negative dimensions are not allowed`` if more than
					   # 2 peaks are detected, then increase to 3
		resonances = np.concatenate([s11_mins, np.zeros(exp_n_res+1-s11_mins.shape[0])])
	return resonances


def check_sampling_frequency(files):
	fs_str = files[0].parts[-2].split('_')[-1]
	fs_int = int(fs_str.split('Hz')[0])
	start = read.read_timestamp_vna(files[0].stem)
	end = read.read_timestamp_vna(files[-1].stem)
	t_total = (end-start).total_seconds()
	fs_calc = len(files)/t_total
	print(f'\nNominal sampling frequency = {fs_int} Hz'
	   f'\tCalculated sampling frequency = {fs_calc:0.2f} Hz')
	

def main():
	# directories and files
	meas_instr = 'VNA'
	test_group = interactive.choose_files(Path().absolute().parent.parent/meas_instr/'data')
	# directory with files
	directory = Path().absolute().parent.parent/meas_instr/'data'/test_group
	# if further subdirectories exist, parse them 
	if any(entry.is_dir() for entry in directory.iterdir()):
		subdir_name = interactive.choose_files(directory)
		subdir = Path().absolute().parent.parent/meas_instr/'data'/test_group/subdir_name
		# directories to save plots and results
		plots_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group/subdir_name
		res_dir = Path().absolute().parent.parent/meas_instr/'results'/test_group
		res_file = Path(res_dir/f'{subdir_name}.csv')
	else:
		subdir = directory
		# directories to save plots and results
		plots_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group
		res_dir = Path().absolute().parent.parent/meas_instr/'results'
		res_dir.mkdir(parents=True, exist_ok=True)
		res_file = Path(res_dir/f'{test_group}.csv')
	if not plots_dir.is_dir():
		plots_dir.mkdir(parents=True, exist_ok=True)
	if not res_dir.is_dir():
		res_dir.mkdir(parents=True, exist_ok=True)
	# Look for data files (different formats)
	csv_files = list(subdir.glob('*.csv'))
	s1p_files = list(subdir.glob('*.s1p'))
	s1p_format = 1 # by default assume files are in touchstone format (s1p)
	if csv_files:
		files = csv_files
		s1p_format = 0
	elif s1p_files:
		files = s1p_files

	# specify desired frequency range for the plots if previously known
	f_range = interactive.set_frequency_range_plots()

	# initialize time column (time calculated from timestamp in filename)
	timestamps = []
	# loop through files and run analysis
	if not res_file.is_file(): # only run once, i.e. if result file does not exist
		print(f'Analyzing {test_group}')
		# ask the user if they want to save one plot for each file 
		save_plots = interactive.choose_to_save_plots(len(files))
		# initialize variables
		names = [] # list of test names
		arrs = [] # list of resonance frequencies 
		print(f'Processing {len(files)} files')
		# Check sampling frequency based on the number of files and total test time
		if 'Hz' in test_group: # only for directories containing sampling frequency in the name
			check_sampling_frequency(sorted(files))

		for i, fp in enumerate(sorted(files)):
			file_name = PurePath(fp).name.split('.')[0]
			t = read.read_timestamp_vna(file_name)
			timestamps.append(t)
			names.append(file_name)
			# Read differently based on format
			if s1p_format == 0:
				f, Z, ph, s11_mag_db, s11_ph = read.read_vna_csv(fp)
			else:
				f, Z, ph, s11_mag_db, s11_ph = read.read_vna_s1p(fp)
			# Find resonance frequency based on S11 minima and append to results
			exp_n_res = 2
			arr = utilities.find_f_res(f, Z, ph, s11_mag_db, exp_n_res)
			arrs.append(arr)
			# Plot one figure for each file only if required by the user
			if save_plots:
				# create directory to store plots
				plots_dir.mkdir(parents=True, exist_ok=True)
				# plot S11 and Z 
				# specify desired frequency range for the plots if previously known
				fig = plot.plot_S11_and_Z(f, Z, ph, s11_mag_db, s11_ph, file_name, f_range)
				fig.savefig(plots_dir/f'{file_name}.png', bbox_inches='tight')
				print(f'Figure saved to \t{plots_dir}/{file_name}.png')
		
		# save results as dataframe
		cols = [f'f{i+1} (S11) (MHz)' for i in range(exp_n_res)] # e.g. for 2 sensors
																 # 'f1 (S11) (MHz)', 'f2 (S11) (MHz)'
		df = pd.DataFrame(arrs, columns=cols)
		df.insert(0, 'Test name', names)
		ts = [(t-timestamps[0]).total_seconds() for t in timestamps]
		df.insert(1, 'Time (s)', ts)
		df.to_csv(res_file, index=False)
		print(f'Results saved to {res_file}')


if __name__ == '__main__':
	main()