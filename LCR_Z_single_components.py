'''Plot data from LCR files.

Usage: launch the script from Anaconda prompt or command prompt (Windows)
	conda activate wireless_multisensing
	cd /d <directory_containing_code>
	python LCR_Z_single_components.py

'''
import pandas as pd
from pathlib import Path
import numpy as np
import utils.plot as plot
import utils.utilities as utilities
import utils.interactive as interactive
import utils.read as read


def main():
	# globals

	# dictionary containing initial sensor length needed for strain calculation 
	sensor_length = {'C2': 50, 'Csin3': 50, 'Csin3_shieldex': 50, 'C1,2': 30, 
				  'Csin3_stitched': 50, 'Csin3_stitched-1': 50,'Csin4':60, 
				  'Csin4-1':60, 'Csin4-2':60, 'Csin5': 70, 'C10':60, 'Cr4': 60,
				  'Chm6_shieldex':70} 
	# dictionary containing number of preconditioning cycles (varies with test) 
	# # TO DO improvement: track it automatically based on hold time after preconditioning
	n_precon_cycles = {'C10': 3, 'C1,2': 3,  'Csin4-1': 0, 'Csin5': 0, 'Csin4-2': 5, 
					'Cr4':5, 'Csin3_stitched': 5, 'Csin3_stitched-1': 5, 'Chm6_shieldex':5}
	test_group = interactive.choose_files(Path().absolute().parent.parent/'LCR'/'data')
	
	# directories with data
	directory = Path().absolute().parent.parent/'LCR'/'data'/test_group
	# if subdirectory exist, go to that
	if any(entry.is_dir() for entry in Path(directory).iterdir()):
		subdir_name = interactive.choose_files(Path(directory))
		directory = Path().absolute().parent.parent/'LCR'/'data'/test_group/subdir_name
	files = list(directory.glob('**/*.tsv')) # both single files and files in subfolders
	
	# directories to save plots and results
	plots_dir = Path().absolute().parent.parent/'LCR'/'plots'/test_group/subdir_name
	plots_dir.mkdir(parents=True, exist_ok=True)
	res_dir = Path().absolute().parent.parent/'LCR'/'results'/test_group
	res_dir.mkdir(parents=True, exist_ok=True)
	res_fp = Path(res_dir/f'{subdir_name}.csv')

	# initialize variables for results storing
	disps = [] # applied displacement
	nom_strain = [] # nominal strain (based on sensor initial length, not the real strain)
	C0s = [] # baseline capacitance
	Cmaxs = []
	deltaC_abs = [] # absolute change in capacitance
	deltaC_pc = [] # percentage change in capacitance

	for fp in files[1:]:
		name = fp.parts[-2]+'_'+fp.stem
		
		print(f'\nReading {fp}')
		df = read.read_tsv(fp)
		t = df['Time (s)'].values
		try:
			C = df['CS'].values/1e-12 # convert to pF
		except KeyError:
			C = df['CP'].values/1e-12 # convert to pF
		cutoff = 2 # cutoff frequency for low pass filter, Hz 
		C_filt = utilities.lp_butter(t, C, cutoff) # low pass butterworth
		C = C_filt
		if 'zwick' not in test_group:
			# Plot capacitance only
			fig = plot.plot_C_LCR(t, C_filt, title=fp.stem)
			fig.savefig(plots_dir/f'{name}.png', bbox_inches='tight')
			print(f'\nSaved capacitance plot to {plots_dir}/{name}.png')
			continue
		# if strain data available (instron / zwick roell) plot both
		elif 'zwick' in test_group:
			fp_strain = fp.parent/'strain_data'/f'{fp.stem}.TRA'
			df_strain = read.read_utm(fp_strain) # read file from UTM (Zwick-Roell)
			disp_from_name = int(fp.stem.split('mm')[0])
			disps.append(disp_from_name)
			t_strain = df_strain['Test time'].values # time stamps
			load = df_strain['Standard force'].values # load
			disp = df_strain['Grip to grip separat'].values # displacement
			disp -= disp[0] # relative displacement
			l0 = sensor_length[subdir_name] # mm, initial length of the sample 
			strain = utilities.calculate_strain(disp, l0)
			nom_strain.append(disp_from_name/l0*100)

			# synchronize using first peak
			t_strain_new = utilities.sync_strain_c(t_strain, strain, t, C)

			if 'Csin5' in subdir_name:
				# remove the first cycle (wrong)
				id_start = np.argmin(np.abs(t_strain_new-8.0))
				print(id_start, t_strain_new[id_start])
				t_strain_new1 = t_strain_new[id_start:]
				print(len(t_strain_new), len(t_strain_new1))
				t_strain_new = t_strain_new1
				disp = disp[id_start:]
			
			t_c_id_start = np.argmin(np.abs(t-t_strain_new[0]))
			t_c_id_end = np.argmin(np.abs(t-t_strain_new[-1]))
			t_C = t[t_c_id_start:t_c_id_end]
			C = C[t_c_id_start:t_c_id_end]
			
			# remove preconditioning cycle if present
			if n_precon_cycles[subdir_name] != 0 :
				t_strain, disp, t_C, C = utilities.remove_precond_cycles(t_strain_new, disp, 
																t_C, C, n_precon_cycles[subdir_name])
			else:
				t_strain = t_strain_new

			# optional: plot strain and load
			# fig = plot.plot_strain_and_load(t_strain_new, strain, load)		
			# fig.savefig(plots_dir/f'{disp_from_name}mm_strain_vs_load.png', bbox_inches='tight')
			# print(f'\nSaved plot to {plots_dir}/{disp_from_name}mm_strain_vs_load.png')

			# Calculate percentage change in capacitance to store 
			C0, Cmax, dC_abs, dC_pc = utilities.calculate_deltaC(C)
			C0s.append(round(C0, 2))
			Cmaxs.append(round(Cmax, 2))
			deltaC_abs.append(dC_abs)
			deltaC_pc.append(dC_pc)
			# Plot
			fig = plot.plot_disp_and_C(t_strain, disp, t_C, C, dC_pc)
			fig.savefig(plots_dir/f'{disp_from_name}mm.png', bbox_inches='tight')
			print(f'\nSaved plot to {plots_dir}/{disp_from_name}mm.png')
		
		# Save dataframe with results
		res_df = pd.DataFrame(list(zip(disps, nom_strain, C0s, Cmaxs, deltaC_abs, deltaC_pc)), 
						columns=['Displacement (mm)', 'Nominal strain (%)', 
			   			'Baseline C (pF)', 'Peak C (pF)', 'ΔC (pF)', 'ΔC (%)'])
		res_df.to_csv(res_fp, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
	main()