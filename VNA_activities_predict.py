'''Evaluate classification model performance for previously stored features and trained models
(performed in VNA_activities_classify.py).

the user can choose a specific classification model or models are iterated from a predefined
list corresponding to the trained models. The models are fit on separate test data (not 
used during training phase) and the corresponding classification metrics reported with scores,
classification report and plots (confusion matrix and true vs predicted values).

Usage: launch the script from Anaconda prompt or command prompt (Windows):

	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python VNA_activities_predict.py

'''
from pathlib import Path
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.interactive as interactive
import utils.classification as classification
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set to avoid very small number with scientific notation when operation yields 0
np.set_printoptions(precision=4, suppress=True)

# Global variables
meas_instr = 'VNA'
data_dir = Path('Z:/2024_Wireless sensing multiple sensors/Data/2024_Wireless Sensing Multiple Sensors/VNA/data')
# directory with classification data
cls_dir_all = data_dir.absolute().parent.parent/meas_instr/'classification'
test_group = interactive.choose_files(cls_dir_all)
cls_dir = Path().absolute().parent.parent/meas_instr/'classification'/test_group
cls_dir.mkdir(parents=True, exist_ok=True)
# directories to save plots
plots_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group
plots_dir.mkdir(parents=True, exist_ok=True)
plots_cls_dir = Path().absolute().parent.parent/meas_instr/'plots'/test_group/'classification'
plots_cls_dir.mkdir(parents=True, exist_ok=True)
# Activities mapping after removing stand and transition phases
static_act = {0: 'deep squat', 1: 'foot to glute', 2: 'hip abduction', 
			  3: 'hip flexion', 4: 'knee drive', 5: 'squat'}
dynamic_act = {0: 'deep squat', 1: 'foot to glute', 2: 'hip abduction', 
			   3: 'hip flexion', 4: 'knee drive', 5: 'back lunge R', 
			   6: 'back lunge L', 7: 'run', 8: 'squat', 9: 'walk'}


def main():
	# Potential models for activity classification 
	model_names_list = ['knn', 'logreg', 'svc', 'xgb', 'dectree', 'rf']
	# choose model for activity classification
	# model_name = interactive.choose_from_options(model_names_list)
	# run analysis on static, dynamic or all activities together
	activity_types = ['static', 'dynamic']
	activity_type = interactive.choose_from_options(activity_types)
	if activity_type == 'static':
		class_map = static_act
	elif activity_type == 'dynamic':
		class_map = dynamic_act

	# Load data (previously stored feature dataset)
	feat_fp = Path(cls_dir/f'features_{activity_type}.csv')
	df_features = pd.read_csv(feat_fp)
	# plot.plot_features(df_features)
	X = df_features.drop(columns=['y'], axis=1)
	y = df_features['y']
	classification.feature_inspection(df_features)
	EXIT()
	
	# split in training validation and test based on number of cycles for each activity
	X_train, y_train, X_val, y_val, X_test, y_test, cycle_ranges_dict = classification.train_test_split_per_cycle(X, y)
	
	# Remove standing and transition phases 
	X_train_fit, y_train_fit, mask_train_fit = classification.remove_transition_and_standing(X_train, y_train)
	X_test_fit, y_test_fit, mask_test_fit = classification.remove_transition_and_standing(X_test, y_test)
	# Encode labels (start from 0, needed for some models e.g. xgboost)

	le = LabelEncoder()
	le.fit(y_train_fit)
	y_test_enc = le.transform(y_test_fit)
	y_train_enc = le.transform(y_train_fit)

	model_names = [] 
	accuracies = []
	f1_scores = []
	clf_report_dfs = []

	for model_name in model_names_list:
		print(model_name)
		model_names.append(model_name)
		# Load trained model
		trained_model_fp = Path(cls_dir/'trained_models'/f'{model_name}_{activity_type}.pickle')
		trained_model = pickle.load(open(trained_model_fp, 'rb'))
		print(f'\nModel was already trained\nLoading model from: {trained_model_fp}')

		# Use trained model_name on test data
		scores_test, clf_report, fig_cm, fig_true_pred = classification.predict_on_test(X_test_fit, y_test_enc, 
																trained_model, class_map) 
		# report scores and append for global reporting (all models)
		print(f'Test scores:\t{scores_test}')
		accuracies.append(scores_test[0])
		f1_scores.append(scores_test[1])
		# Save classification report
		clf_report_fp = Path(cls_dir/'clf_reports'/f'{model_name}_{activity_type}_test.csv')
		clf_report.to_csv(clf_report_fp, index=True)
		clf_report_dfs.append(clf_report)
		# Save confusion matrix figure
		fig_cm.savefig(plots_cls_dir/'confusion_matrices'/f'{activity_type}_{model_name}.png', bbox_inches='tight')
		# Save figure with true and predicted values
		fig_name = f'{model_name}_{activity_type}.png'
		fig_true_pred.savefig(plots_cls_dir/'true_vs_pred'/fig_name, bbox_inches='tight')
		

	# Save scores in single dataframe for all models
	df = pd.DataFrame(list(zip(model_names, accuracies, f1_scores)), columns=['model', ' accuracy', 'f1 score'])
	df.to_csv(cls_dir/f'{activity_type}_classification_results.csv', index=False)
	# Save classification reports (scores per class) in single dataframe for all models
	df_clf_reports = pd.concat(clf_report_dfs, axis=0)
	print(df_clf_reports)
	
if __name__ == '__main__':
	main()