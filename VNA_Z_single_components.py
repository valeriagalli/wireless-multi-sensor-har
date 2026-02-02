"""Plot impedance data from the VNA file.

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	
	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python VNA_Z_single_components.py

"""
import pandas as pd
from pathlib import Path, PurePath
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.read as read
import utils.interactive as interactive
from matplotlib import pyplot as plt


def check_sampling_frequency(files):
	fs_str = files[0].parts[-2].split("_")[-1]
	fs_int = int(fs_str.split("Hz")[0])
	start = read.read_timestamp_vna(files[0].stem)
	end = read.read_timestamp_vna(files[-1].stem)
	t_total = (end-start).total_seconds()
	fs_calc = len(files)/t_total
	print(f"\nNominal sampling frequency = {fs_int} Hz"
	   f"\tCalculated sampling frequency = {fs_calc:0.2f} Hz")
	

def main():
	# directories and files
	meas_instr = "VNA"
	test_group = interactive.choose_files(Path().absolute().parent.parent/meas_instr/"data")
	if test_group != "Z_single_components":
		print("WARNING: using wrong module, this module is for calculating" \
		"impedance of single circuit components. Exiting...")
		exit()
	# directory with files
	directory = Path().absolute().parent.parent/meas_instr/"data"/test_group
	# if further subdirectories exist, parse them 
	if any(entry.is_dir() for entry in directory.iterdir()):
		subdir_name = interactive.choose_files(directory)
		subdir = Path().absolute().parent.parent/meas_instr/"data"/test_group/subdir_name
		# directories to save plots and results
		plots_dir = Path().absolute().parent.parent/meas_instr/"plots"/test_group/subdir_name
		res_dir = Path().absolute().parent.parent/meas_instr/"results"/test_group
		res_file = Path(res_dir/f"{subdir_name}.csv")
	else:
		subdir = directory
		# directories to save plots and results
		plots_dir = Path().absolute().parent.parent/meas_instr/"plots"/test_group
		res_dir = Path().absolute().parent.parent/meas_instr/"results"
		res_dir.mkdir(parents=True, exist_ok=True)
		res_file = Path(res_dir/f"{test_group}.csv")
	if not plots_dir.is_dir():
		plots_dir.mkdir(parents=True, exist_ok=True)
	if not res_dir.is_dir():
		res_dir.mkdir(parents=True, exist_ok=True)
	# Look for data files (different formats)
	csv_files = list(subdir.glob("*.csv"))
	s1p_files = list(subdir.glob("*.s1p"))
	s1p_format = 1 # by default assume files are in touchstone format (s1p)
	if csv_files:
		files = csv_files
		s1p_format = 0 # set to 0 if files are actually in csv format 
	elif s1p_files:
		files = s1p_files

	# specify desired frequency range for the plots if previously known
	f_range = interactive.set_frequency_range_plots()
	
	# loop through files and run analysis
	if not res_file.is_file(): # only run once, i.e. if result file does not exist
		print(f"Analyzing {test_group}")
		save_plots = 1 
		# initialize variables
		names = [] # list of test names
		self_res = [] # self resonance of each component 
		print(f"Processing {len(files)} files")
		# Check sampling frequency based on the number of files and total test time
		if "Hz" in test_group: # only for directories containing sampling frequency in the name
			check_sampling_frequency(sorted(files))
		for fp in sorted(files):
			file_name = PurePath(fp).name.split(".")[0]
			names.append(file_name)
			# Read differently based on format
			if s1p_format == 0:
				f, Z, ph, s11_mag_db, s11_ph = read.read_vna_csv(fp)
				f_self = f[np.argmax(Z)]
				self_res.append(f_self)
			else:
				f, Z, ph, s11_mag_db, s11_ph = read.read_vna_s1p(fp)
			# Plot one figure for each file only if required by the user
			if save_plots:
				# create directory to store plots
				plots_dir.mkdir(parents=True, exist_ok=True)
				# plot S11 and Z
				fig = plot.plot_S11_and_Z(f, Z, ph, s11_mag_db, s11_ph, file_name, f_range)
				fig.savefig(plots_dir/f"{file_name}.png", bbox_inches="tight")
				print(f"Figure saved to \t{plots_dir}/{file_name}.png")
		
		# save results as dataframe
		df = pd.DataFrame(self_res, columns=["f_self (MHz)"])
		df.insert(0, "Test name", names)
		df.to_csv(res_file, index=False)
		print(f"Results saved to {res_file}")


if __name__ == "__main__":
	main()