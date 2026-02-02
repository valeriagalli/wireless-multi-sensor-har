'''Utility functions to get input from command line.'''


def choose_files(directory):
	'''Choose a specific file or folder from a directory. 
	Args:
		directory (Path): Input directory.
	Returns:
		choice (str): Chosen folder or file.
	'''
	choices = sorted([f for f in list(directory.iterdir()) if f.is_dir()]) # or Path(f).suffix=='.CSV']
	print('\nChoose from:')
	print('\n0 - all')
	for i, n in enumerate(choices):
		nn = n.name.split('/')[-1]
		print(f'{i+1} - {nn}')
	index = int(input('\nType the index: '))
	choice = choices[index-1].name.split('/')[-1]
	print(f'\nChosen file or folder --> {choice}')
	return choice


def choose_meas_instr(directory):
	'''Choose a specific file or folder from a directory. 
	Args:
		directory (Path): Input directory.
	Returns:
		choice (str): Chosen folder or file.
	'''
	choices = list(directory.iterdir())
	print('\nChoose from:')
	for i, n in enumerate(choices):
		nn = n.name.split('/')[-1]
		print(f'{i} - {nn}')
	index = int(input('\nType the index: '))
	choice = choices[index].name.split('/')[-1]
	print(f'\nChosen measurement instrument --> {choice}')
	return choice


def choose_to_analyze_s11():
	'''Choose whether to use S11 or not (to identify resonance frequency).	 
	Returns:
		choice (bool): yes (1) or no (0).
	'''
	choice = int(input('Include S11 in the analysis?\nYES --> press 1\nNO --> press 0\n'))
	return choice


def choose_to_save_plots(n_files):
	'''Choose whether to save or not a plot.
	Args:
		n_files (int): number of files for which to save plots	 
	Returns:
		choice (bool): yes (1) or no (0).
	'''
	choice = int(input(f'Save one figure for each file ({n_files} files)?\nYES --> press 1\nNO --> press 0\n'))
	return choice


def set_res_frequency_range_plots():
	# specify desired frequency range for the plots if previously known
	set_f_range = int(input('\nSpecify expected resonance frequency range for y-axis in plots?'
						 '\n0 - NO\n1 - YES\n'))
	if set_f_range:
		fmin = float(input('Expected minimum resonance frequency (MHz):\t'))
		fmax = float(input('Expected maximum resonance frequency (MHz):\t'))
		f_range = [fmin, fmax]
	else:
		f_range = None
	return f_range


def choose_from_options(options_list):
	print('\nChoose from:')
	for i, name in enumerate(options_list):
		print(f'{i} - {name}')
	index = int(input('\nType the index: '))
	choice = options_list[index]
	print(f'\nChosen --> {choice}')
	return choice