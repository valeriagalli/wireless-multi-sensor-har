'''Analyze and plot data from VNA files recorded during activities with prototype. All 
s1p touchstone files for a specific test (one file per frequency sweep) are read and 
for each file the resonance frequencies corresponding to each sensor are stored. The 
result is a csv file containing one row per file (i.e. one per sweep) with the file name, 
timestamps, and the resonance frequencies. 

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python VNA_activities_preprocess.py

'''
import re
import pandas as pd
from pathlib import Path, PurePath
import numpy as np
from matplotlib import pyplot as plt
import utils.plot as plot
import utils.utilities as utilities
import utils.read as read
import utils.interactive as interactive

def main():
	# directories and files
	meas_instr = 'VNA'
	data_dir = Path('Z:/2024_Wireless sensing multiple sensors/Data/2024_Wireless Sensing Multiple Sensors/VNA/data') # confidential data 
	test_group = interactive.choose_files(data_dir)
	directory = data_dir/test_group
	activities = [x for x in directory.iterdir() if x.is_dir()]

	# directories to save plots and results
	res_dir = Path().absolute().parent.parent/meas_instr/'results'/test_group
	res_dir.mkdir(parents=True, exist_ok=True)
	plots_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group
	plots_dir.mkdir(parents=True, exist_ok=True)

	# specify desired resonance frequency range for the plots if previously known
	f_res_range = interactive.set_res_frequency_range_plots()

	range_dict = {} # dictionary containing ranges of f1 and f2 for each activity
	dominant_f1 = []
	dominant_f2 = []

	# loop through files
	for activity in activities: # one csv file per activity
		# Make subdirectories
		activity_name = str(activity.stem)
		res_file = Path(res_dir/f'{activity_name}.csv')
		print(f'\nTest group = {test_group} ---- Activity = {activity_name}\n')

		if not res_file.is_file(): # only run the analysis once
			files = list(activity.glob('*.s1p'))
			if not files:
				print(f'WARNING: no files in folder {activity}')

			# ask the user if they want to save one plot for each file 
			save_plots = 0 # interactive.choose_to_save_plots(len(files))
			if save_plots:
				# specify desired resonance frequency range for the plots if previously known
				f_range = interactive.set_frequency_range_plots()
			# Define expected number of resonance frequencies (e.g. default 2 for 2 sensors)
			exp_n_res = 2

			# Empty arrays to store data
			timestamps = [] # list for time stamps
			names = [] # list of test names
			arrs = [] # list of resonance frequencies 

			print(f'Reading {len(files)} files...')
			# Loop through single files (one per sweep)
			for fp in sorted(files):
				file_name = PurePath(fp).name.split('.')[0]
				f, Z, ph, s11_mag_db, s11_ph = read.read_vna_s1p(fp)			
				# Find resonance frequency from various signal features
				try:
					arr = utilities.find_f_res(f, Z, ph, s11_mag_db, exp_n_res)
				except ValueError:
					print(f'WARNING no s11 peaks detected for {fp}')
					continue
				arrs.append(arr)
				names.append(file_name)
				# Timestamp 
				t = read.read_timestamp_vna(fp.stem)
				timestamps.append(t)
				# plot and save S11 and impedance
				if save_plots:
					fig = plot.plot_S11_and_Z(f, Z, ph, s11_mag_db, s11_ph, file_name, f_range=None)
					fig.savefig(plots_dir/activity/f'{file_name}.png', bbox_inches='tight')
					print(f'Figure saved to \t{plots_dir}/{file_name}.png')

			# save results as dataframe
			cols = [f'f{i+1} (S11) (MHz)' for i in range(exp_n_res)] # e.g. for 2 sensors
																	# 'f1 (S11) (MHz)', 'f2 (S11) (MHz)'
			df = pd.DataFrame(arrs, columns=cols)
			df.insert(0, 'Test name', names)
			ts = [(t-timestamps[0]).total_seconds() for t in timestamps]
			df.insert(1, 'Time (s)', ts)
			df['f2 (S11) (MHz)'] = df['f2 (S11) (MHz)'].where(df['f3 (S11) (MHz)'] == 0, df['f3 (S11) (MHz)'])
			df.to_csv(res_file, index=False)
			print(f'Results saved to:\t {res_file}')

		else: # if results already exist, load and plot results
			# print(f'\nAnalysis already run, extracting results from: \n{res_file}')
			df = pd.read_csv(res_file)
			stats = df.describe()
			range_row = stats.iloc[7,:]-stats.iloc[3, :] # calculate range (max-min)
			# optional: save full statistics dataframe
			# stats = pd.concat([stats, pd.DataFrame([range_row])], ignore_index=True)
			# stats_col = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max','range']
			# stats.insert(0, 'statistic', stats_col) 
			# stats.to_csv(Path(res_dir/f'{activity_name}_STATS.csv'), index=False)
			range_dict[activity_name] = range_row
			# dominant signal frequency
			t = df['Time (s)']
			f1 = df['f1 (S11) (MHz)']
			f2 = df['f2 (S11) (MHz)']
			print('Dominant frequency based on sensor 1:')
			dominant_f1.append(utilities.find_dominant_f(t, f1))
			print('Dominant frequency based on sensor 2:')
			dominant_f2.append(utilities.find_dominant_f(t, f2))
			# Plot data for specific activity
			# lab1 = r'$C_1$'
			# lab2 = r'$C_2$'
			# title = activity_name
			# fig_res = plot.plot_results_activities(t, f1, f2, lab1, lab2, title, f_res_range)
			# fig_res.savefig(plots_dir/f'{activity_name}.png', bbox_inches='tight')
			# fig_res_delta = plot.plot_results_activities_delta_f(t, f1, f2, lab1, lab2, 
			# 										   title)
			# fig_res_ratio = plot.plot_results_activities_ratio_f(t, f1, f2, lab1, lab2, 
			# 										   title)
			# fig_res_delta.savefig(plots_dir/f'{activity_name}_delta_f.png', bbox_inches='tight')
			# fig_res_ratio.savefig(plots_dir/f'{activity_name}_ratio_f.png', bbox_inches='tight')

	# ranges_df = pd.DataFrame.from_dict(range_dict)
	# ranges_df.to_csv(res_dir/'ranges'/'ranges_all.csv')
	dominant_f_df = pd.DataFrame(list(zip(dominant_f1, dominant_f2)), 
							  columns=['dominant f1 (Hz)', 'dominant f2 (Hz)'])
	activity_names_list = [a.stem for a in activities]
	dominant_f_df.insert(0, 'Activity', activity_names_list)
	print(dominant_f_df)
	dominant_f_df.round(2).to_csv(res_dir/'dominant_frequency_all.csv', index=False)

if __name__ == '__main__':
	main()