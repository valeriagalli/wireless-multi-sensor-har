
'''Plot data from impedance analyzer files, either saved from the instrument (csv) or from
labview for continuous saving.

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	
	conda activate wireless_multisensing 
	cd /d <directory_containing_code>
	python IA_Z_single_components.py

'''
import pandas as pd
from pathlib import Path
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.interactive as interactive
import utils.read as read


def main():
	test_group = interactive.choose_files(Path().absolute().parent.parent/'IA'/'data'/'WIRED_EXPERIMENTS'/'Z_single_components')
	# directory with data
	directory = Path().absolute().parent.parent/'IA'/'data'/'WIRED_EXPERIMENTS'/'Z_single_components'/test_group
	subdir_name = interactive.choose_files(Path(directory))
	if subdir_name == 'all':
		files = list(subsubdir.glob('**/*.csv')) 
	else:
		subdir = Path(directory/subdir_name)
		if any([d.is_dir() for d in list(subdir.iterdir())]):
			subsubdir_name = interactive.choose_files(Path(subdir))
			subsubdir = Path(subdir/subsubdir_name)
			# directories to save plots and results
			plots_dir = Path().absolute().parent.parent/'IA'/'plots'/'WIRED_EXPERIMENTS'/'Z_single_components'/test_group/subdir_name
			res_dir = Path().absolute().parent.parent/'IA'/'results'/'WIRED_EXPERIMENTS'/'Z_single_components'/test_group/subdir_name
		else:
			subsubdir = subdir
			subsubdir_name = subdir_name
			# directories to save plots and results
			plots_dir = Path().absolute().parent.parent/'IA'/'plots'/'WIRED_EXPERIMENTS'/'Z_single_components'/test_group
			res_dir = Path().absolute().parent.parent/'IA'/'results'/'WIRED_EXPERIMENTS'/'Z_single_components'/test_group
		files = list(subsubdir.glob('*.csv')) 
	

	# create directories if they don't exist already 
	if not plots_dir.is_dir():
		plots_dir.mkdir(parents=True, exist_ok=True)
	if not res_dir.is_dir():
		res_dir.mkdir(parents=True, exist_ok=True)

	# Create dataframe and empty lists to save results
	comp_names = []
	f_self_values = [] 
	models = []
	r2_values = [] 
	mean_values = []

	# specify desired frequency range for the plots if previously known
	f_range = interactive.set_frequency_range_plots()

	for fp in sorted(files):
		name = fp.parts[-2]+'_'+fp.stem
		
		ia_labview = 0 # by default assume measurement was done with instrument (vs labview application)
		print('##########################################################################')
		print(f'Reading {fp}')
		try:
			df, f, Z, ph = read.read_imp_an(fp)
			if 'Rs[ohm]' in df.columns:
				R = df['Rs[ohm]'].values
			elif 'Rp[ohm]' in df.columns:
				R = df['Rp[ohm]'].values
			elif 'X[ohm]' in df.columns:
				R =  df['X[ohm]'].values
		except KeyError:
			ia_labview = 1
			df, f, Z, ph, R = read.read_imp_an_labview(fp)
		# upsample 
		if ia_labview == 1:
			up_factor = 20 # based on the number of sweep points (e.g. with Labview app fewer points are used to speed up meas)
		else:
			up_factor = 5
		f1, Z1 = utilities.upsample(f, Z, up_factor)
		f1, ph1 = utilities.upsample(f, ph, up_factor)
		f1, R1 = utilities.upsample(f, R, up_factor)
		
		# Inductors
		if 'L' in name:
			if ia_labview == 1 :
				L = df['LS'].values[1:]/1e-6 # remove NaN from empty row (already implemented in ``read_imp_an_labview``)
			else:
				try:
					L = df['Ls[H]'].values/1e-6 
				except KeyError:
					L = df['Lp[H]'].values/1e-6 
			f1, L1 = utilities.upsample(f, L, up_factor)
			# L below self resonance (f < f_selfres/2)
			i = utilities.find_sign_inversion(L1) # self resonance (f_selfres)
			idx = int(len(f)/2)
			if i.any():
				idx = i[0] # self resonance (f_selfres), typically only 1 in freq range choice
			print('sign inversion', idx) 
			model, r2 = utilities.quadratic_fit(f1[0:int(idx/2)], L1[0:int(idx/2)])
			coefs = model.coef
			model_str = f'{coefs[0]:.3e} x2 + {coefs[1]:.3e} x + {coefs[2]:.3f}'
			mean = np.mean(L1[0:int(idx/2)])

			print('mean inductance', mean)
			print('model string', model_str)
			# Plot
			title = fp.stem
			fig = plot.plot_L(f1, L1, idx, model, r2, title, f_range)
			fig.savefig(plots_dir/f'{name}.png', bbox_inches='tight')
			print(f'\nSaved inductance plot to {plots_dir}/{name}.png')
			fig1 = plot.plot_L_and_Z(f1, Z1, ph1, R1, L1, title, f_range)
			fig1.savefig(plots_dir/f'{name}_L_and_Z.png', bbox_inches='tight')
			print(f'\nSaved inductance plot to {plots_dir}/{name}_L_and_Z.png')
		# Capacitors
		elif 'C' in name:
			if ia_labview == 1:
				C = df['CP'].values[1:]/1e-12 # remove NaN from empty row (already implemented in ``read_imp_an_labview``)
			else:
				try:
					C = df['Cp[F]'].values/1e-12 
				except KeyError:
					C = df['Cs[F]'].values/1e-12 
			f1, C1 = utilities.upsample(f, C, up_factor)
			f1, R1 = utilities.upsample(f, R, up_factor)
			# C below self resonance (f < f_selfres/2)
			i = utilities.find_sign_inversion(C1) # self resonance (f_selfres)
			idx = int(len(f)/2)
			if i.any():
				idx = i[0] # self resonance (f_selfres), typically only 1 in freq range choice 
			model, r2 = utilities.quadratic_fit(f1[0:int(idx/2)], C1[0:int(idx/2)])
			coefs = model.coef
			model_str = f'{coefs[0]:.3e} x2 + {coefs[1]:.3e} x + {coefs[2]:.3f}'
			mean = np.mean(C1[0:int(idx/2)])
			# Plot
			title = fp.stem
			fig = plot.plot_C(f1, C1, idx, model, r2, title, f_range)
			fig.savefig(plots_dir/f'{name}.png', bbox_inches='tight')
			print(f'\nSaved capacitance plot to {plots_dir}/{name}.png')
			fig1 = plot.plot_C_and_Z(f1, Z1, ph1, R1, C1, title, f_range)
			fig1.savefig(plots_dir/f'{name}_C_and_Z.png', bbox_inches='tight')
			print(f'\nSaved inductance plot to {plots_dir}/{name}_C_and_Z.png')

		# Save results to csv file
		comp_names.append(name)
		f_self_values.append(f1[idx])
		models.append(model_str)
		r2_values.append(r2)
		mean_values.append(mean)
		
		df = pd.DataFrame(list(zip(comp_names, f_self_values, models, r2_values, mean_values)),
							columns=['Component', 'f_self (MHz)', 'Model (f<f_self/2)', 'R2 (f<f_self/2)', 'Mean (f<f_self/2)'])

		df.to_csv(res_dir/f'{subsubdir_name}.csv', index=False, float_format='%.3f')
		print(f'\nSaved results file to  {res_dir}/{subsubdir_name}.csv')

	# join all results in single file
	print(f'\nGenerate single result file for group {subsubdir_name}?')
	reply = str(input('[y] / [n]\n'))
	if reply == 'y':
		res_files = [f for f in list(res_dir.glob('*.csv')) if 'all' not in f.name]
		dfs = []
		for f in res_files:
			df = pd.read_csv(f)
			dfs.append(df)
		df_all = pd.concat(dfs)
		df_all.to_csv(res_dir/'all.csv', index=False)


if __name__ == '__main__':
	main()