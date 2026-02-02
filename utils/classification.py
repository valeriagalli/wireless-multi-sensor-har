'''Utility functions for classification tasks.'''

import itertools
import numpy as np
import pandas as pd
import math
from scipy import stats, signal
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import learning_curve, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from utils import plot
from utils import utilities
from matplotlib import pyplot as plt

# Global variables
RANDSTATE = 42 # keep random state consistent throughout algorithms
TRAIN_SIZE = 0.8
VAL_SIZE = 0.10
TEST_SIZE = 0.10


############### PREPROCESSING ###############
def normalize(X):
	'''Normalize feature data. 
	Note: scaling may not effect some models (e.g. Random Forest Regressor) but others (Neural Network or linear models).
	Args:
		X (arr): Features.
	Returns:
		Xnorm (arr): Normalized features.
	'''
	sc_X = StandardScaler()
	X_norm = sc_X.fit_transform(X)
	return X_norm


def label_cycles(t, f, tol_low, tol_high, cycle_duration):
	'''
	Detect standing vs activity phases in signal (e.g. for squat test: standing 5s, then squat 5s etc).
	Thresholding based on minimum and maximum resonance frequencies with a tolerance.
	This retains the transition phases between stand and activities such that the resulting
	indices arrays have the same length as the original data.

	Args:
		t (arr): Timestamps.
		f (arr): Resonance frequency (signal of interest).
		tol_low (float): Tolerance for low thresholding (+- tol*(frequency range).
		tol_high (float): Tolerance for high thresholding (+- tol*(frequency range).
	Returns:
		label (arr): Array of indices identifying the phase: 0 for standing, 
							1 for activity and -1 for transitions.
	'''
	f_sample = 1/np.mean(np.diff(t))
	t = np.asarray(t, dtype=float).reshape(-1)
	f = np.asarray(f, dtype=float).reshape(-1)
	f_min, f_max = np.min(f), np.max(f)
	low_thresh = f_min + tol_low * (f_max - f_min)
	high_thresh = f_max - tol_high * (f_max - f_min)
	labels = []
	for i in range(len(f)):
		if f[i] < low_thresh: # activity phase
			labels.append(1)
		elif f[i] > high_thresh: # standing phase
			labels.append(0)
		else: # transition phase
			labels.append(-1)
	min_transition_duration = round(cycle_duration/4*f_sample) # in nr. samples
	labels_corrected = correct_labels(np.array(labels), min_transition_duration)
	return labels_corrected


def correct_labels(labels, min_transition_duration):
	'''
	Corrects labels for a cyclic labels with +1, 0, -1: +1 activity, -1 transition, 0 standing (rest).
	- Short transition (-1) spikes <= min_transition_duration are absorbed into surrounding +1/0 plateaus.
	- Real transitions (-1) (> min_transition_duration) are preserved.
	
	Args:
		labels (arr): Initially identified labels to be corrected  (+1, 0, -1).
		min_transition_duration: maximum length of -1 considered a spike to absorb
	
	Returns:
		corrected (arr): corrected labels (+1, 0, -1).
	'''
	n = len(labels)
	corrected = np.copy(labels)
	# Step 1: detect runs
	run_starts = [0]
	run_values = [labels[0]]
	for i in range(1, n):
		if labels[i] != labels[i-1]:
			run_starts.append(i)
			run_values.append(labels[i])
	run_starts.append(n)
	# Step 2: merge short -1 spikes
	for r in range(1, len(run_values)-1):
		val = run_values[r]
		start = run_starts[r]
		end = run_starts[r+1]
		length = end - start
		if val == -1 and length <= min_transition_duration:
			prev_val = run_values[r-1]
			next_val = run_values[r+1]
			if prev_val == next_val and prev_val in [0,1]:
				# merge the spike into surrounding plateau
				corrected[start:end] = prev_val
	return corrected


def merge_labels(labels1, labels2):
	'''
	Merge two label arrays into a consensus (from the two sensors).
	Activity = 1, Stand = 0, Transition = -1.
	Whenever the labels do not correspond, label as transition phase (-1), not used in 
	classification. This sacrifices part of the data but ensures no confusion between
	actual phases (stand vs activity) during classification.
	'''
	labels1 = np.array(labels1)
	labels2 = np.array(labels2)
	consensus = np.full_like(labels1, -1)  # start with all transitions
	for i in range(len(labels1)):
		if labels1[i] == labels2[i]:
			# both agree → keep label
			consensus[i] = labels1[i]
		elif (labels1[i] in [0,1]) and (labels2[i]==-1):
			# one says stand or activity (e.g. squat), the other transition → stand or activity
			consensus[i] = labels1[i]
		elif (labels1[i] == -1) and (labels2[i] in [0, 1]):
			 # one says stand or activity (e.g. squat), the other transition → stand or activity
			consensus[i] = labels2[i]
	return consensus


def feature_extraction(X, y, ws):
	'''overlapping windows with 80% overlap'''
	# preserve activity name for easier data handling 
	activity_name = X['Activity']
	step_size = round(ws/5)
	# features
	mean1 = []
	sd1 = []
	var1 = []
	min1 = []
	max1 = []
	mean2 = []
	sd2 = []
	var2 = []
	min2 = []
	max2 = []
	y_mode = []
	activity_name_win = []
	f1 = X['f1'].values
	f2 = X['f2'].values
	for i in range(0, X.shape[0]-ws+1, step_size):
		win1 = f1[i:i+ws] # first sensor
		mean1.append(np.mean(win1))
		sd1.append(np.std(win1)) 
		var1.append(np.var(win1))
		min1.append(np.min(win1))
		max1.append(np.max(win1))
		win2 = f2[i:i+ws] # second sensor
		mean2.append(np.mean(win2)) 
		sd2.append(np.std(win2)) 
		var2.append(np.var(win2))
		min2.append(np.min(win2))
		max2.append(np.max(win2))
		y_mode.append(stats.mode(y[i:i+ws], keepdims=False).mode) # most frequent label in the window
		activity_name_win.append(Counter(activity_name[i:i+ws]).most_common(1)[0][0])
	cols = ['mean1', 'sd1', 'var1', 'min1', 'max1', 'mean2', 'sd2', 'var2', 'min2', 'max2', 
		 'y', 'Activity']
	df_features = pd.DataFrame(list(zip(mean1, sd1, var1, min1, max1,
									 mean2, sd2, var2, min2, max2,
									 y_mode, activity_name_win)), 
									 columns=cols)
	return df_features


def spectral_features(sig, fs):
	f, Pxx = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))  # Power Spectral Density
	Pxx_norm = Pxx / np.sum(Pxx)  # normalize power for entropy & centroid
	dom_freq = f[np.argmax(Pxx)]  # dominant frequency
	centroid = np.sum(f * Pxx_norm)  # spectral centroid
	if np.sum(Pxx) == 0:
		Pxx_norm = np.ones_like(Pxx) / len(Pxx)
	else:
		Pxx_norm = Pxx / np.sum(Pxx)
	spec_entropy = stats.entropy(Pxx_norm)
	return dom_freq, centroid, spec_entropy


def feature_extraction_expanded(X, y, fs, window_s, overlap):
	'''
	Sliding window feature extraction preserving all short transitions.

	Any window containing a transition (-1) will be labeled as transition.
	'''
	f1 = X['f1'].values
	f2 = X['f2'].values
	labels = y
	activities = X['activity'].values

	step = int(window_s * fs * (1 - overlap))
	window_len = int(window_s * fs)
	features_list = []

	for start in range(0, len(f1) - window_len + 1, step):
		end = start + window_len
		seg1 = f1[start:end]
		seg2 = f2[start:end]
		seg_label = labels[start:end]
		seg_activity = activities[start:end]

		dom_f1, cent_f1, ent_f1 = spectral_features(seg1, fs)
		dom_f2, cent_f2, ent_f2 = spectral_features(seg2, fs)

		# --- compute features ---
		feats = {
			'mean_f1': np.mean(seg1),
			'mean_f2': np.mean(seg2),
			'std_f1': np.std(seg1),
			'std_f2': np.std(seg2),
			'rms_f1': np.sqrt(np.mean(seg1**2)),
			'rms_f2': np.sqrt(np.mean(seg2**2)),
			'range_f1': np.max(seg1) - np.min(seg1),
			'range_f2':np.max(seg2) - np.min(seg2),
			# 'slope_f1': (seg1[-1] - seg1[0]) / (len(seg1)/fs),
			# 'slope_f2': (seg2[-1] - seg2[0]) / (len(seg2)/fs),
			# 'mean_der_f1': np.mean(np.diff(seg1)),
			# 'mean_der_f2': np.mean(np.diff(seg2)),
			# 'std_der_f1': np.std(np.diff(seg1)),
			# 'std_der_f2': np.std(np.diff(seg2)),
			'corr_f1_f2': np.corrcoef(seg1, seg2)[0,1],
			# 'energy_ratio_f1_f2': np.sum(seg1**2)/np.sum(seg2**2),
			# 'skew_f1': stats.skew(seg1),
			# 'skew_f2': stats.skew(seg2),
			# 'kurt_f1': stats.kurtosis(seg1),
			# 'kurt_f2': stats.kurtosis(seg2),
			# 'dom_freq_f1': dom_f1,
			# 'dom_freq_f2': dom_f2,
			# 'centroid_f1': cent_f1,
			# 'centroid_f2': cent_f2,
			'entropy_f1': ent_f1,
			'entropy_f2': ent_f2,
		}

		# --- assign window label ---
		if -1 in seg_label:
			feats['y'] = -1  # preserve any transition in the window
		else:
			vals, counts_no_trans = np.unique(seg_label, return_counts=True)
			feats['y'] = vals[np.argmax(counts_no_trans)]

		# --- assign activity name ---
		feats['activity'] = pd.Series(seg_activity).mode()[0]
		features_list.append(feats)
	return pd.DataFrame(features_list)


def feature_inspection(df_features):
	'''Inspect features to discard useless ones.'''
	df = df_features.drop(['activity'], axis=1) # only works with numbers
	sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap=plt.cm.Blues)
	plt.title('Correlation among features')
	# plt.show()
	X = df_features.drop(columns=['y', 'activity'])
	X = X.fillna(X.mean())
	y = df_features['y']
	print(X.isna().sum())
	# ANOVA F-test
	f_vals, p_vals = f_classif(X, y)
	df_f = pd.DataFrame({'feature': X.columns, 'f_score': f_vals, 'p_value': p_vals})
	# Mutual Information
	mi_vals = mutual_info_classif(X, y)
	df_mi = pd.DataFrame({'feature': X.columns, 'mutual_info': mi_vals})
	df_ranked = df_f.merge(df_mi, on='feature').sort_values('f_score', ascending=False)
	return df_ranked


def train_test_split_per_cycle(X, y):
	'''Split into train validation and test set based on activity and number of cycles
	performed for each activity, i.e. full cycles only included in each set.'''
	X_train = []
	y_train = []
	X_val = []
	y_val = []
	X_test = []
	y_test = []
	activity_names = X['Activity']
	act_end_id = 0
	cycle_ranges_dict = {}
	for i, act in enumerate(sorted(set(activity_names))):
		# split by activity and take fist 80% of data for training, 10% for validation, 
		# and 10% for testing
		mask = (X['Activity']==act)
		idx = X.index[mask]
		X_act = X.iloc[idx, :] # features for activity ``act`` only (e.g. ``squat``)
		y_act = y[idx] # labels for activity ``act`` only
		# find start indices for each cycle: from transition (-1) to activity id (+1, +2, etc), first activity datapoint (+1)
		cycle_start_ids_act = np.where(np.diff(y_act)==max(y_act)+1)[0] + 1
		# find end indices for each cycle: from standing (0) to transition (-1) 
		cycle_end_ids_act = np.where(np.diff(y_act)==-1)[0][1:] # discard the first 
		cycle_end_ids_act = np.append(cycle_end_ids_act, len(y_act)-1) # add end point
		cycle_start_ids = act_end_id + cycle_start_ids_act  # convert to global indices
		cycle_end_ids = act_end_id + cycle_end_ids_act  # convert to global indices
		cycle_ranges = [(int(s), int(e)) for s, e in zip(cycle_start_ids, cycle_end_ids)]
		cycle_ranges_dict[i] = cycle_ranges
		n_samples = len(y_act)
		n_cycles_act = len(cycle_start_ids_act)
		# Update act_end_id for next activity
		act_end_id += n_samples
		# optional: plot cycles start and end points for visual check
		# plot.plot_cycle_starts_ends(X_act, y_act, cycle_start_ids_act, cycle_start_ids, cycle_end_ids_act, cycle_end_ids)
		# plt.show()
		n_cycles_val = math.ceil(VAL_SIZE*n_cycles_act)
		n_cycles_test = math.ceil(TEST_SIZE*n_cycles_act)
		n_cycles_train = n_cycles_act-n_cycles_val-n_cycles_test
		# Optional: print out number of cycles (total, train, val and test)
		# print(f'\nActivity {act} --> labels = ({set(y_act)})')
		# print(f'Total: {n_cycles_act} cycles\nTrain: {n_cycles_train} cycles'
		# 	  f'\nValidation: {n_cycles_val}\nTest: {n_cycles_test}')
		train_start = cycle_start_ids_act[0] # discard initial standing phase_test_start = cycle_start_ids[-2]
		test_start =  cycle_start_ids_act[-n_cycles_test]
		val_start = cycle_start_ids_act[-(n_cycles_test+n_cycles_val)]
		X_train_act = X_act.iloc[train_start:val_start, :]
		y_train_act = y_act[train_start:val_start]
		X_val_act = X_act.iloc[val_start:test_start, :]
		y_val_act = y_act[val_start:test_start]
		X_test_act = X_act.iloc[test_start:, :]
		y_test_act = y_act[test_start:]
		# optional: plot train test validation split for visual check
		# plot.plot_train_val_test_split(X_act, y_act, X_train_act, y_train_act, X_val_act, y_val_act,
		# 						 X_test_act, y_test_act)
		X_train.append(X_train_act)
		y_train.append(y_train_act)
		X_val.append(X_val_act)
		y_val.append(y_val_act)
		X_test.append(X_test_act)
		y_test.append(y_test_act)
	# From list of arrays (one array per activity) to single arrays
	X_train_arr = np.vstack(X_train)
	y_train_arr = np.array(list(itertools.chain.from_iterable(y_train)))
	X_val_arr = np.vstack(X_val)
	y_val_arr = np.array(list(itertools.chain.from_iterable(y_val)))
	X_test_arr = np.vstack(X_test)
	y_test_arr = np.array(list(itertools.chain.from_iterable(y_test)))
	# Trasform in dataframes and drop the activity names (not needed anymore)
	X_train_df = pd.DataFrame(X_train_arr, columns = X.columns).drop(columns=['Activity'])
	X_val_df = pd.DataFrame(X_val_arr, columns = X.columns).drop(columns=['Activity'])
	X_test_df = pd.DataFrame(X_test_arr, columns = X.columns).drop(columns=['Activity'])
	X_train = X_train_df
	X_val =  X_val_df
	X_test = X_test_df
	# Ensure numpy 1D arrays
	y_train = y_train_arr.astype(int).ravel()
	y_val   = y_val_arr.astype(int).ravel()
	y_test  = y_test_arr.astype(int).ravel()
	# Scale features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train) 
	X_val_scaled = scaler.transform(X_val)   
	X_test_scaled = scaler.transform(X_test) 
	# optional check 
	print('full dataset class counts:\n', pd.Series(y).value_counts().sort_index())
	print('Training set class counts:\n', pd.Series(y_train).value_counts().sort_index())
	print('Validation set class counts\n:', pd.Series(y_val).value_counts().sort_index())
	print('Test set class counts:\n', pd.Series(y_test).value_counts().sort_index())
	return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, cycle_ranges_dict


def remove_transition_and_standing(X, y, ignored_classes=[-1,0]):
	'''
	Keep all samples in X, y but return a filtered version for training.
	
	Parameters
	----------
	X : pandas.DataFrame or np.ndarray
		Feature matrix
	y : np.ndarray
		Label array
	ignored_classes : list
		Classes to ignore for training (default: [-1, 0])
	
	Returns
	-------
	X_fit : same type as X
		Features for training (ignoring transition/standing)
	y_fit : np.ndarray
		Labels for training (ignoring transition/standing)
	mask_fit : np.ndarray
		Boolean mask of which samples were used for training
	'''
	mask_fit = ~np.isin(y, ignored_classes)
	y_fit = y[mask_fit]
	if isinstance(X, np.ndarray):
		X_fit = X[mask_fit]
	else:  # pandas DataFrame
		X_fit = X.iloc[mask_fit].reset_index(drop=True)
	# optional: visual check
	fig = plot.plot_removed_standing_and_transition(X, X_fit, mask_fit)
	return X_fit, y_fit, mask_fit


############### MODEL DEFINITION ###############
def model_def(model_name):
	models_dict = {
		'knn': KNeighborsClassifier(),
		'logreg': LogisticRegression(),
		'svc': SVC(),
		'xgb': XGBClassifier(),
		'dectree': DecisionTreeClassifier(),
		'rf': RandomForestClassifier(),
		}
	return models_dict[model_name]


def param_def(model_name):
	params_knn = {
		'n_neighbors': range(1, 20), 
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'brute']}
	params_logreg = {
		'C': [1, 10, 100],
		'penalty': ['l2'],
		'solver': ['lbfgs'],  # saga works with both l1/l2 and multinomial
		'max_iter': [5000],
		'class_weight': ['balanced'],
		'random_state': [RANDSTATE]
	}
	params_svc = {
		'C': [0.1, 1, 10, 100],
		'kernel': ['linear', 'rbf', 'poly'],
		'gamma': ['scale', 'auto', 0.01, 0.1, 1],
		'degree': [2, 3, 4],  # Only for 'poly' kernel
		'class_weight': ['balanced'],
		}
	params_xgb = {
		'n_jobs': [-1],
		'n_estimators': [100, 200],
		'learning_rate': [0.01, 0.1, 0.2],
		'max_depth': [2,4,6],
		'subsample': [0.5, 0.9],
		'random_state': [RANDSTATE],
		}
	params_dectree = {
		'criterion': ['log_loss', 'entropy', 'gini'],
		'max_depth': [2,4,6],
		'min_samples_leaf':[3,5,7],
		'min_samples_split':[8,10,12],
		'class_weight': ['balanced'],
		'random_state': [RANDSTATE],
		}
	params_rf = {
		'max_depth': [2,4,6], 
		'n_estimators': [10,20,50],
		'min_samples_split': [2,5,7,10],
		'min_samples_leaf': [1,2,5], 
		'class_weight': ['balanced'],
		'random_state': [RANDSTATE],
		}
	params_dict = {'knn': params_knn, 'logreg': params_logreg, 'svc': params_svc, 'xgb': params_xgb,
				   'rf': params_rf, 'dectree': params_dectree}
	return params_dict[model_name]


def model_build_default(model_name):
	models_dict = {
		'knn': KNeighborsClassifier(), 
		'logreg': LogisticRegression(random_state=RANDSTATE), 
		'svc': SVC(random_state=RANDSTATE),
		'xgb': XGBClassifier(random_state=RANDSTATE), 
		'dectree': DecisionTreeClassifier(random_state=RANDSTATE), 
		'rf': RandomForestClassifier(random_state=RANDSTATE)
		}
	return models_dict[model_name]


def model_build(model_name, params):
	# print('\nUsing model --{model_name}-- with the following parameters:\n{params}')
	if model_name == 'knn':
		model = KNeighborsClassifier(**params)
	elif model_name == 'logreg':
		model = LogisticRegression(**params)
	elif model_name == 'svc':
		model = SVC(**params)
	elif model_name == 'rf':
		model = RandomForestClassifier(**params)
	elif model_name == 'dectree':
		model = DecisionTreeClassifier(**params)
	elif model_name == 'xgb':
		model = XGBClassifier(**params)
	return model


############### TRAINING ###############
def check_classes_in_fold(cv, X_train, y_train):
	'''Check that each fold contains all classes and the number of samples per class.'''
	for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
		train_classes = np.unique(y_train[train_idx])
		test_classes = np.unique(y_train[test_idx])
		print(f'Fold {fold_idx + 1}:')
		# Print unique classes
		print(f'  Train classes: {train_classes}')
		print(f'  Test classes:  {test_classes}')
		# Optional: check if any class is missing in training
		missing_classes = set(np.unique(y_train)) - set(train_classes)
		if missing_classes:
			print(f'  Missing classes in training: {missing_classes}')
		# Count samples per class in train and test
		print("  Train samples per class:")
		for cls in train_classes:
			print(f'    Class {cls}: {(y_train[train_idx] == cls).sum()}')
		print("  Test samples per class:")
		for cls in test_classes:
			print(f'    Class {cls}: {(y_train[test_idx] == cls).sum()}')


def hyperpar_tuning(model_name, X_train, y_train):
	'''Hyperparameter tuning.'''
	print(f'\nTuning hyperparameters for {model_name}')
	params = param_def(model_name)
	model = model_def(model_name)
	# define splits for cross validation
	cv_strat=StratifiedKFold(n_splits=4, shuffle=True)
	# optional: check that all folds contain all classes and the corresponding count
	check_classes_in_fold(cv_strat, X_train, y_train)
	# Use splits in GridSearchCV
	grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv_strat,
							verbose=True, error_score=np.nan)
	grid_search.fit(X_train, y_train)
	best_par = grid_search.best_params_
	print(f'Best parameters for {model_name} --> {best_par}')
	return best_par


def evaluate_bias_variance_tradeoff(model_name, model, X_train, y_train):
	train_sizes, train_scores, val_scores = learning_curve(
			model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1,
			train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDSTATE)
	train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
	val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)
	fig_learning_curve = plot.plot_learning_curve(model_name, train_sizes, train_mean, train_std, val_mean, val_std)
	return fig_learning_curve
   

def train_model(X_train, y_train, X_val, y_val, model_name, class_map):
	'''Implements algorithm on train and validation datasets, returns best model over cross validaiton.
	Args:
		X_train (Dataframe): Training set (features).
		y_train (arr): Training set (Labels).
		X_val (Dataframe): Validation set (features).
		y_val (arr): Validation set (labels).
		model_name (str): Name of the model to be used for the prediction, e.g. Random Forest Regressor.
		class_map (dict): Mapping of activity names.
	Returns:
		model (pickle): Trained model.
	'''
	# Remove standing and transition phases from training set
	X_train_fit, y_train_fit, mask_train_fit = remove_transition_and_standing(X_train, y_train)
	# Encode labels (start from 0 instead of 1, needed for some models e.g. xgboost)
	le = LabelEncoder()
	le.fit(y_train_fit)
	y_train_enc = le.transform(y_train_fit)
	# Tune hyperparameters 
	best_params = hyperpar_tuning(model_name, X_train_fit, y_train_enc) 
	model = model_build(model_name, best_params)
	fig_learning_curve = evaluate_bias_variance_tradeoff(model_name, model, X_train_fit, y_train_enc)
	# Use cross validation to find best model 
	scoring = 'f1_weighted' # better than accuracy for imbalanced classes
	print(f'\nCross validation with 4 folds on training set (scoring = {scoring})')
	cv_strat=StratifiedKFold(n_splits=4, shuffle=True)
	check_classes_in_fold(cv_strat, X_train_fit, y_train_enc)
	scores = cross_validate(model, X_train_fit, y_train_enc, scoring=scoring, 
						 return_estimator=True, return_train_score=True, cv=cv_strat)
	scores_train = scores['train_score']
	scores_test = scores['test_score']
	print(f'Cross validation scores on TRAINING set:') 
	print(f'{scoring} - Training set: {scores_train}')
	print(f'{scoring} - Test set: {scores_test}')
	# Pick best model
	ii = np.argmax(abs(scores['test_score'])) # max F1 indicates best performance
	print(f'Model with best score = {ii}')
	best_model = scores['estimator'][ii]
	# Predict on validation set
	# Remove standing and transition phases from validation set
	X_val_fit, y_val_fit, mask_val_fit = remove_transition_and_standing(X_val, y_val)
	# Encode labels (start from 0, needed for some models e.g. xgboost)
	y_val_enc = le.transform(y_val_fit)
	y_pred = best_model.predict(X_val_fit)
	# Classification report on validation set
	print('Classification Report')
	clf_report_dict = classification_report(y_val_enc, y_pred, digits=3, 
										 output_dict=True, labels=list(class_map.keys()),
										 target_names=list(class_map.values()))
	df_report = pd.DataFrame(clf_report_dict).transpose() # Transposing puts classes/labels as rows and metrics as columns
	# Plot Confusion matrix on validation set
	cm = confusion_matrix(y_val_enc, y_pred)
	fig_cm = plot.plot_confusion_matrix(cm, list(class_map.values()))
	# Plot True vs predicted values on validation set
	fig = plot.plot_true_pred(y_val_enc, y_pred, 'Validation set')
	return best_model, df_report, fig_cm, fig_learning_curve


############### TESTING AND EVALUATION ###############
def predict_on_test(X_test, y_test, trained_model, class_map):
	'''Load trained model and predict  on unseen test data.
	Args:
		X_test (Series): Feature test set (transition and standing removed).
		y_test (arr): Encoded true labels for the test set (transition and standing removed).
		trained_model (pickle.obj): previously trained model.
		class_map (dict): Mapping of activity names.
	'''
	# (standing and transition phases previously removed)
	# Predict on test set
	y_pred = trained_model.predict(X_test)
	# optional check: make sure all classes are present 
	check_all_classes_included(y_test, y_pred)
	# Calculate scores (accuracy and F1-score)
	scores = calculate_scores(y_test, y_pred)
	# Classification report on test set
	clf_report_dict = classification_report(y_test, y_pred, 
										 digits=3, output_dict=True, 
										 labels=list(class_map.keys()),
										 target_names=list(class_map.values()))
	df_report = pd.DataFrame(clf_report_dict).transpose() # Transposing puts classes/labels as rows and metrics as columns
	# Plot Confusion matrix  on test set
	cm = confusion_matrix(y_test, y_pred)
	fig_cm = plot.plot_confusion_matrix(cm, list(class_map.values()))
	# PlotTrue vs predicted values on test set
	fig_true_pred = plot.plot_true_pred(y_test, y_pred, 'Test set')
	return scores, df_report, fig_cm, fig_true_pred


def calculate_scores(y_true, y_pred):
	acc = round(accuracy_score(y_true, y_pred), 2)
	f1 = round(f1_score(y_true, y_pred, average='weighted'), 2) 
	scores = [acc, f1]
	print(f'Accuracy = {acc}\tWeighted F1-score = {f1}')
	return scores


def check_all_classes_included(y_true, y_pred):
	"""	Ensuring all classes are included in the model training or evaluation and prints out
	the count for each class.

	Args:
	y_true (arr): Encoded true labels (after LabelEncoder).
	y_pred (arr): Encoded predicted labels.
	"""
	# Ensure numeric numpy arrays
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	# Check class coverage
	print("\nLabel coverage check:")
	print(f"  Unique in y_true: {np.unique(y_true)}\tCounts in y_true: {Counter(y_true)}")
	print(f"  Unique in y_pred: {np.unique(y_pred)}\tCounts in y_pred: {Counter(y_pred)}")

