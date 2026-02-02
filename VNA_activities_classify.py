'''Prepare data from preprocessed VNA files recorded during activities for activity classification 
and train classification models.

Preprocessed data (from s1p files for single sweeps to csv files with resonance frequencies 
in time) is previously generated with the module ''VNA_activities_preprocess.py''.

This module lets the user choose which test to analyze (i.e. which participant, which session)
then preprocesses data from ativities and prepares a dataset for classification (labbelled)
or if the dataset for classification was already created applies a classification algorithm 
to predict the activity.

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	
	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python VNA_activities_classify.py

'''
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.interactive as interactive
import utils.classification as classification
from matplotlib import pyplot as plt

# Set to avoid very small number with scientific notation when operation yields 0
np.set_printoptions(precision=4, suppress=True)

# Global variables
meas_instr = 'VNA'
data_dir = Path('Z:/2024_Wireless sensing multiple sensors/Data/2024_Wireless Sensing Multiple Sensors/VNA/data')
# previously analyzed data in results files (from VNA_activities_preprocess.py)
res_dir_all = data_dir.absolute().parent.parent/meas_instr/'results'
test_group = interactive.choose_files(res_dir_all)
res_dir = Path().absolute().parent.parent/meas_instr/'results'/test_group
res_files_all = [f for f in res_dir.glob('*.csv') if 'ranges' not in f.name]
res_files_static = [f for f in res_files_all if 'dyn' not in f.name and 'stand' not in f.name]
res_files_dyn    = [f for f in res_files_all if 'dyn' in f.name and 'stand' not in f.name]
# directories to save plots
plots_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group
plots_dir.mkdir(parents=True, exist_ok=True)
plots_cls_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group/'classification'
plots_cls_dir.mkdir(parents=True, exist_ok=True)
# directory for classification data
cls_dir = Path().absolute().parent.parent/meas_instr/'classification'/test_group
cls_dir.mkdir(parents=True, exist_ok=True)
# Models for activity classification 
model_names_list = ['knn', 'logreg', 'svc', 'xgb', 'dectree', 'rf']
# Activities mapping after removing stand and transition phases
static_act = {0: 'deep squat', 1: 'foot to glute', 2: 'hip abduction', 
			  3: 'hip flexion', 4: 'knee drive', 5: 'squat'}
dynamic_act = {0: 'deep squat', 1: 'foot to glute', 2: 'hip abduction', 
			   3: 'hip flexion', 4: 'knee drive', 5: 'back lunge R', 
			   6: 'back lunge L', 7: 'run', 8: 'squat', 9: 'walk'}


def label_data(f, id_act, lp_cutoff, duration, tol_low, tol_high):
	'''Prepare classification dataset for specific test: label activity phases in the test.
	For example, if the test contains 5 cycles of an activity (3.g. squat) of ca 5 seconds 
	each: label the first 5 seconds as ``stand``, then the next 5 seconds as ``squat``, 
	and so on.
	
	Args:
		f (Path): File path of the original dataframe containing the data for a test, 
					i.e. timestamps and values of resonance frequency 1 (sensor 1) and 
					resonance frequency 2 (sensor 2).
		id_act (int): Activity id.
		lp_cutoff (float): Cutoff frequency for low pass filtering.
		duration (float): Duration of each cycle, 5 seconds for static tests. 
							For dynamic test this can vary from 0.5 to 1.5 seconds 
							depending on the activity.
		tol_low (float): tolerance for theshold of lower resonance frequency (activity phases).
		tol_high (float): tolerance for theshold of higher resonance frequency (stand phases).

	Returns:
		df_labelled (Dataframe): Labelled dataset, i.e. with added columns identifying the
						label as an integer: 0 for ``stand``, -1 for transitions between
						stand and activity, x for ``activity`` where x varies based on 
						the activity.
	'''
	df = pd.read_csv(f, usecols=['Time (s)', 'f1 (S11) (MHz)', 'f2 (S11) (MHz)'])		
	print(f'\nReading results from:\t{f}')
	t = df['Time (s)'].values
	fs = 1/(np.mean(np.diff(t)))
	print(f'Sampling frequency = {fs:.2f} Hz')
	f1 = df['f1 (S11) (MHz)'].values # sensor 1 knee
	f1_norm = f1-np.mean(f1[0:10])
	f2 = df['f2 (S11) (MHz)'].values # sensor 2 glute
	f2_norm = f2-np.mean(f2[0:10])
	# Low pass filter(for static activities)
	f1_filt = utilities.lp_butter(t, f1, lp_cutoff)
	f2_filt = utilities.lp_butter(t, f2, lp_cutoff)
	f1_norm_filt = utilities.lp_butter(t, f1_norm, lp_cutoff)
	f2_norm_filt = utilities.lp_butter(t, f2_norm, lp_cutoff)	
	print(f'Low pass filtered data with cutoff: {lp_cutoff} Hz')
	# tolerances for detecting cycles of stand vs activity
	tol_low = 0.15 # activity phases (sensors stretched)
	tol_high = 0.20 # stand phases usually more noisy
	# Label each state: stand (0), transition (-1) or activity (id_act, integer > 0)
	labels1 = classification.label_cycles(t, f1_norm_filt, tol_low, tol_high, duration) # Sensor 1
	labels2 = classification.label_cycles(t, f2_norm_filt, tol_low, tol_high, duration) # Sensor 2 
	# Merge conflicts between labels of sensor 1 and sensor 2
	labels_merged = classification.merge_labels(labels1, labels2)
	# Saved in labelled dataset 
	# Add normalized and filtered frequency columns 
	df['f1_norm_filt'] = f1_norm_filt
	df['f2_norm_filt'] = f2_norm_filt
	# Add label columns (identify stand phase vs transition phase vs activity phase)
	# to distinguish among activities, multiply by ``id_act`` which changes with each 
	# result file (i.e. each recorded activity).
	df['label_sensor1'] = np.array(labels1)*(id_act+1) # 1st id_act would be 0, add 1 because 0 is for standing phases
	df['label_sensor2'] = np.array(labels2)*(id_act+1)
	df['label_merged'] = np.array(labels_merged)*(id_act+1)
	df['Activity'] = f.stem
	# Replace all transition phases with -1 (instead of different negative number per activity)
	df['label_sensor1'] = df['label_sensor1'].where(df['label_sensor1'] >= 0, -1)
	df['label_sensor2'] = df['label_sensor2'].where(df['label_sensor2'] >= 0, -1)
	df['label_merged'] = df['label_merged'].where(df['label_merged'] >= 0, -1)
	# Plot labelled data for visual check 
	fig = plot.plot_activity_labelled(t, f1_filt, labels1, f2_filt, labels2, title=f.stem)
	fig.savefig(plots_cls_dir/'labelled'/f'{f.stem}_labelled_both.png', bbox_inches='tight')
	fig = plot.plot_activity_merged_labels(t, f1_filt, f2_filt, labels_merged, title=f.stem)
	fig.savefig(plots_cls_dir/'labelled'/f'{f.stem}_labelled_merged.png', bbox_inches='tight')
	return df


def calculate_average_delta_f(activity_name, df):
	# Calculate average delta f for each sensor (average across cycles)
	mean_f1_stand = df.loc[df['label_sensor1'] == 0, 'f1 (S11) (MHz)'].mean()
	sd_f1_stand = df.loc[df['label_sensor1'] == 0, 'f1 (S11) (MHz)'].std()
	mean_f1_act = df.loc[df['label_sensor1'] > 0, 'f1 (S11) (MHz)'].mean()
	sd_f1_act = df.loc[df['label_sensor1'] > 0, 'f1 (S11) (MHz)'].std()
	range_f1 = mean_f1_stand - mean_f1_act
	mean_f2_stand = df.loc[df['label_sensor2'] == 0, 'f2 (S11) (MHz)'].mean()
	sd_f2_stand = df.loc[df['label_sensor2'] == 0, 'f2 (S11) (MHz)'].std()
	sd_f2_act = df.loc[df['label_sensor2'] > 0, 'f2 (S11) (MHz)'].std()
	mean_f2_act = df.loc[df['label_sensor2'] > 0, 'f2 (S11) (MHz)'].mean()
	range_f2 = mean_f2_stand - mean_f2_act
	keys = ['Activity', 'mean_f1_stand', 'sd_f1_stand,', 'mean_f1_act', 'sd_f1_act', 'delta f1',
				     		'mean_f2_stand', 'sd_f2_stand,', 'mean_f2_act', 'sd_f2_act','delta f2']
	df_ranges = pd.DataFrame([[activity_name, mean_f1_stand, sd_f1_stand, mean_f1_act, sd_f1_act, range_f1, 
									 mean_f2_stand, sd_f2_stand, mean_f2_act, sd_f2_act, range_f2]], 
									 columns = ['Activity', 'mean_f1_stand', 'sd_f1_stand,', 'mean_f1_act', 'sd_f1_act', 'delta f1',
				     		'mean_f2_stand', 'sd_f2_stand,', 'mean_f2_act', 'sd_f2_act','delta f2'])
	return df_ranges


def set_parameters_based_on_activity_type(activity_name):
	if 'dyn' in activity_name:
		tol_high = 0.20
		tol_low = 0.15
		lp_cutoff = 5
		cycle_duration = 1
		win_size = 0.2
		if 'run' in activity_name:
			cycle_duration = 0.5
			win_size = 0.1
	else: 
		tol_high = 0.15
		tol_low = 0.10
		lp_cutoff = 1 # Hz
		cycle_duration = 5 # seconds
		win_size = 1 # seconds
	return lp_cutoff, cycle_duration, win_size, tol_low, tol_high


def main():
	# run analysis on static, dynamic or all activities together
	activity_types = ['static', 'dynamic']
	activity_type = interactive.choose_from_options(activity_types)
	if activity_type == 'static':
		res_files = res_files_static
		class_map = static_act
	elif activity_type == 'dynamic':
		res_files = res_files_dyn
		class_map = dynamic_act
	# dataframe to store all ranges for each sensor 
	dfs_ranges = []

	# If classification dataset not previously saved, preprocess and prepare it
	labelled_fp = Path(cls_dir/f'labelled_data_{activity_type}.csv')
	feat_fp = Path(cls_dir/f'features_{activity_type}.csv')
	if not feat_fp.is_file(): #  if no feature dataset was saved
		if not labelled_fp.is_file(): # if only raw data exist, it has not been labelled 
			print(f'\nPreparing dataset for classification for {len(res_files)} files...')
			# store data in single dataframe containing all the activities 
			# initialize arrays to store data 
			df_all = []
			t_adjusted_all = []
			# pool all data from different activites together
			for id_act, f in enumerate(res_files): 
				lp_cutoff, cycle_duration, _, tol_high, tol_low = set_parameters_based_on_activity_type(f.stem)
				df_labelled = label_data(f, id_act, lp_cutoff, cycle_duration, tol_low, tol_high)
				
				activity_name = f.stem
				df_range = calculate_average_delta_f(activity_name, df_labelled)
				dfs_ranges.append(df_range)

				# adjust timestamp to have consecutive activities not restarting from  t = 0s
				if id_act == 0 :
					t_adjusted = df_labelled['Time (s)'].values
				else:
					t_adjusted = df_labelled['Time (s)'].values + t_adjusted[-1]
				t_adjusted_all.append(t_adjusted)

				# Append to list of labelled dataframes
				df_all.append(df_labelled)

			# organize data in dataframe and export
			df_all_concat = pd.concat(df_all)	
			t_adjusted_flat = [t for t_arr in t_adjusted_all for t in t_arr]
			df_all_concat.insert(1, 'Time adjusted (s)', t_adjusted_flat)
			df_all_concat.to_csv(labelled_fp, index=False)
			print(f'\nLabelled dataset saved to:\t{labelled_fp}')

			# Dataframe with ranges
			ranges_df = pd.concat(dfs_ranges)
			ranges_df.to_csv(Path(res_dir/f'{activity_type}_ranges.csv'), index=False)

		else: # labelled dataset already saved, feature dataset not saved
			# read labelled dataset
			print(f'Reading labbeled data from {labelled_fp} ...') 
			df = pd.read_csv(labelled_fp)
			# detect sampling rate to choose correct window size for feature extraction
			t = df['Time adjusted (s)']
			f_sampling = round(1/np.mean(np.diff(t)))
			# Rename columns for easier handling and clarity
			df = df.rename(columns={'label_merged': 'Label'})
			df = df.rename(columns={'f1_norm_filt': 'f1'})
			df = df.rename(columns={'f2_norm_filt': 'f2'})
			# data and labels
			X = df.drop(['Time adjusted (s)',  'label_sensor1', 'label_sensor2'], axis=1)
			y = df['Label'].values
			# Extract features over moving windows 
			if activity_type != 'all':
				_, _, win_size, _, _ = set_parameters_based_on_activity_type(activity_type)
				ws = round(win_size*f_sampling)
				print(f'\nExtracting features using a window of {win_size} s ({ws} samples)...')
			else:
				df_features = pd.read_csv(feat_fp)
			
			df_features = classification.feature_extraction(X, y, ws)
			feat_fp = Path(cls_dir/f'features_{activity_type}.csv')
			df_features.to_csv(feat_fp, index=False)
			print(f'\nFeature dataset savedt to:\t{feat_fp}')
			# Optional: Visualize features
			classification.feature_inspection(df_features)
	
	# otherwise read feature dataset and apply classification
	else:
		# choose model for activity classification
		model_name = interactive.choose_from_options(model_names_list)
		
		# read dataset
		print(f'Reading feature dataset from:\t{feat_fp} ...') 
		df_features = pd.read_csv(feat_fp)
		# Run classification
		start = time.time() # track computational time

		X = df_features.drop(columns=['y'], axis=1)
		y = df_features['y']		
		activity_names = sorted(set(X['Activity']))
		print('Activities in the chosen dataset:\n', activity_names)

		# Split in training validation and test based on number of cycles for each activity
		X_train, y_train, X_val, y_val, _, _, cycle_ranges_dict = classification.train_test_split_per_cycle(X, y)
		
		# Plot train val test split for visual check
		# fig = plot.plot_train_val_test_split(X, y, X_train, y_train, X_val, y_val, X_test, y_test)
		# fig.savefig(plots_cls_dir/f'{activity_type}_train_test_val_split.png', bbox_inches='tight')

		# If model was already trained and saved, load it
		trained_model_fp = Path(cls_dir/'trained_models'/f'{model_name}_{activity_type}.pickle')

		if 0:# trained_model_fp.is_file():
			trained_model = pickle.load(open(trained_model_fp, 'rb'))
			print(f'\nModel was already trained\nLoading model from: {trained_model_fp}')
		else: # otherwise train and save
			trained_model, clf_report, fig_cm, fig_lc = classification.train_model(X_train, y_train, 
																		X_val, y_val, model_name,
																		class_map) 
			# Save model
			pickle.dump(trained_model, open(trained_model_fp, 'wb'))
			print(f'\n\nSaved trained model to:\t{trained_model_fp}')
			end = time.time()
			print(f'\nTrained {model_name} - execution time = {end-start:.2f} s\n\n')
			# Save classification report
			clf_report_fp = Path(cls_dir/'clf_reports'/f'{model_name}_{activity_type}_train.csv')
			clf_report.to_csv(clf_report_fp, index=True)
			# Save figures for reference
			fig_cm.savefig(plots_cls_dir/'confusion_matrices'/f'{activity_type}_{model_name}.png', bbox_inches='tight')
			fig_lc.savefig(plots_cls_dir/'learning_curves'/f'{activity_type}_{model_name}.png', bbox_inches='tight')

	
if __name__ == '__main__':
	main()