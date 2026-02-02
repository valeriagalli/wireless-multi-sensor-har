'''Utility functions to read data from different instruments and convert formats if needed.'''
import re
import math, cmath
import numpy as np
import pandas as pd
import datetime
import utils.utilities as utilities

##### READ FROM INSTRUMENTS	#####
def read_minivna(fp):
	df = pd.read_excel(fp, skiprows=0)
	f = df['Frequency (Hz)'].values/1e6
	Z = df['|Z| (Ohm)'].values
	ph = df['Theta'].values
	s11_mag = df['Returnloss (dB)'].values
	s11_ph = df['Returnphase (°)'].values
	return f, Z, ph, s11_mag, s11_ph


def read_minivna_autoexport(fp):
	'''Read multiple csv files exported automatically.'''
	df = pd.read_csv(fp, skiprows=0)
	f = df['Frequency(Hz)'].values/1e6
	Z = df['|Z|'].values
	ph = df['Theta'].values
	s11_mag = df['Return Loss(dB)'].values
	s11_ph = df['Phase(deg)'].values
	return f, Z, ph, s11_mag, s11_ph


def read_imp_an(fp):
	''' Read csv file from Impedanca analyzer (saved directly from instrument to USB drive).'''
	with open(fp, 'r') as f:
		lines = f.readlines()
	#discard the rows which do not contain data, find row ``idx`` where data starts (starting with ``No.``)
	idx = 18 # without circuit model in IA file
	for i, l in enumerate(lines):
		if 'FREQ[Hz]' in l:
			idx = i
	df = pd.read_csv(fp, sep=',', skiprows=idx)
	f = df['FREQ[Hz]'].values/1e6
	Z = df['Z[ohm]'].values
	ph = df['PHASE[deg]'].values
	print('\nRead from IA instrument file')
	return df, f, Z, ph


def read_imp_an_labview(fp):
	''' Read csv file from Impedanca analyzer (saved from labview program that controls the IA).'''
	df = pd.read_csv(fp)
	df_list = np.split(df, df[df.isna().all(1)].index) # first row of each df is NaN
	n_sweeps = len(df_list)-1
	print(f'\nNumber of sweeps = {n_sweeps}')
	# for now only take the first sweep
	df0 = df_list[-1] # first one is just header
	try:
		f = df0['Frequency (Hz)'].values[1:]/1e6 # skip the first row (NaN)
	except KeyError:
		f = df0['Frequency'].values[1:]/1e6
	Z = df0['Z'].values[1:]
	ph = df0['PHASE'].values[1:]
	try:
		R = df0['RS'].values[1:]
	except KeyError:
		R = df0['RP'].values[1:]
	print('\nRead from IA labview file')
	return df, f, Z, ph, R


def read_nanovna(fp):
	df = pd.read_csv(fp)
	f = df['Freq (MHz)'].values
	Z_re = df['re(Z)'].values
	Z_im = df['im(Z)'].values
	Z, ph = utilities.re_im_to_mag_ph(Z_re, Z_im)
	s11_re = df['re(S11)'].values
	s11_im = df['im(S11)'].values
	s11_mag, s11_ph = utilities.re_im_to_mag_ph(s11_re, s11_im)
	s11_mag_db = 20*np.log10(s11_mag)
	return f, Z, ph, s11_mag_db, s11_ph


def read_nanovna_cont_export(fp):
	with open(fp, 'r') as f:
		lines = f.readlines()
	headers = ['Freq (MHz)','|Z|', 'arg(Z)', '|S11| dB', 'arg(S11)']
	freq = []
	s11_db = []
	s11_ph = []
	Z_mag = []
	Z_ph = []
	for i,l in enumerate(lines[2:]):
		m = re.search(r'(\d+\.?\d*)\s*(\d+\.?\d*)\s*(\d+\.?\d*)', l)
		freq.append(float(m.group(1)))
		s11_db1 = 20*np.log10(float(m.group(2)))
		s11_db.append(s11_db1)
		s11_ph.append(float(m.group(3)))
		Z_mag1, Z_ph1 = utilities.calculate_Z_from_S11(float(m.group(2)), float(m.group(3)))
		Z_mag.append(Z_mag1)
		Z_ph.append(Z_ph1)
	# df = pd.DataFrame(list(zip(freq, Z_mag, Z_ph, s11_db, s11_ph)), columns=headers)
	return freq, Z_mag, Z_ph, s11_db, s11_ph


def read_nanovna_cont_export_api(fp):
	with open(fp, 'r') as f:
		lines = f.readlines()
	headers = ['', 'Freq (MHz)','S11 mag (dB)', 'S11 arg (deg)']
	sweep_pts = []
	freq = []
	s11_db = []
	s11_ph = []
	Z_mag = []
	Z_ph = []
	for l in lines[1:]:
		items = l.strip().split(',')
		sweep_pts.append(np.int(items[0]))
		freq.append(float(items[1]))
		s11_db.append(float(items[2]))
		s11_ph.append(float(items[3]))
		Z, ph = utilities.calculate_Z_from_S11(utilities.db_to_nat(float(items[2])), float(items[3]))
		Z_mag.append(Z)
		Z_ph.append(ph)
	# df = pd.DataFrame(list(zip(freq, Z_mag, Z_ph, s11_db, s11_ph)), columns=['Freq (MHz)','|Z|', 'arg(Z)', '|S11| dB', 'arg(S11)'])
	return freq, Z_mag, Z_ph, s11_db, s11_ph


def read_tsv(fp, verbose=False):
	with open(fp, 'r') as f:
		lines = f.readlines()
	headers = lines[0].rstrip().split('\t')
	data = dict.fromkeys(headers)
	for n in range(len(headers)):
		data[headers[n]] = [float(l.rstrip().split('\t')[n]) for l in lines[2:]]
	# create dataframe from dictionary
	df = pd.DataFrame.from_dict(data)
	if verbose:
		print(f'Original dataframe: {df.shape}')
	return df


def read_utm(fp):
	'''File from Zwick Roell tensile testing machine.'''
	with open(fp, 'r') as f:
		lines = f.readlines()
	idx = 7 # first line excluding headers
	for i, l in enumerate(lines):
		if 'Time' in l:
			idx = i
	lines_data = lines[idx:]
	headers = lines_data[0].lstrip().split(';')
	data = dict.fromkeys(headers)
	for n in range(len(headers)):
		data[headers[n]] = [float(l.rstrip().split(';')[n]) for l in lines_data[1:]]
	# create dataframe from dictionary
	df = pd.DataFrame.from_dict(data)
	columns = {h: h.strip() for h in df.columns} # delete whitespaces in column names
	df.rename(columns=columns, inplace=True)
	return df


def read_vna_csv(fp):
	'''File from Keysight E5080B benchtop VNA in csv format.'''
	df = pd.read_csv(fp, skiprows=6)
	df = df.iloc[:-1, :] # remove the 'END' row 
	f = df['Freq(Hz)'].values.astype('float')/1e6
	columns = df.columns
	# map_entries = {'S11(MAG)': s11_mag, 'S11(DB)': s11_mag_db,
	# 			'S11(DEG)': s11_ph, 'S11(REAL)': s11_re, 'S11(IMAG)': s11_im} 
	if 'S11(MAG)' in columns and 'S11(DEG)' in columns:
		s11_mag = df['S11(MAG)'].values.astype('float')
		s11_ph = df['S11(DEG)'].values.astype('float')
		s11_mag_db = np.array([20*math.log10(s11_mag_i) for s11_mag_i in s11_mag])
	elif 'S11(DB)' in columns and 'S11(DEG)' in columns:
		s11_mag_db = df['S11(DB)'].values.astype('float')
		s11_ph = df['S11(DEG)'].values.astype('float')
		s11_mag = [10**(float(val_db)/20) for val_db in s11_mag_db]
	elif 'S11(REAL)' in columns and 'S11(IMAG)' in columns:
		s11_re = df['S11(REAL)'].values.astype('float')
		s11_im = df['S11(IMAG)'].values.astype('float')
		s11_mag, s11_ph = utilities.re_im_to_mag_ph(s11_re, s11_im)
		s11_mag_db = np.array([20*math.log10(s11_mag_i) for s11_mag_i in s11_mag])
	Z, ph = utilities.calculate_Z_from_S11(s11_mag, s11_ph)
	return f, Z, ph, s11_mag_db, s11_ph


def read_vna_s1p(fp):
	'''File from Keysight E5080B benchtop VNA in touchstone (s1p) format.'''
	with open(fp, 'r') as f:
		lines = f.readlines()
	freq = []
	s11_db = []
	s11_ph = []
	Z_mag = []
	Z_ph = []
	for i,l in enumerate(lines[1:]):
		nums = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', l)
		numbers = [float(x) for x in nums]  
		freq.append(numbers[0]/1e6) # convert to MHz
		s11_real = numbers[1]
		s11_imag =  numbers[2]
		s11 = complex(s11_real, s11_imag)
		s11_mag = abs(s11)
		s11_phase = math.degrees(cmath.phase(s11))
		s11_db1 = 20*np.log10(s11_mag)
		s11_db.append(s11_db1)
		s11_ph.append(s11_phase)
		Z_mag1, Z_ph1 = utilities.calculate_Z_from_S11(s11_mag, s11_phase)
		Z_mag.append(Z_mag1)
		Z_ph.append(Z_ph1)
	return np.array(freq), np.array(Z_mag), np.array(Z_ph), np.array(s11_db), np.array(s11_ph)


def read_timestamp_minivna(fname):
	'''Read timestamp from miniVNA file. 
	Input: 
		fname (str): filename example: ``VNA_240911_112223``: ``VNA_<yymmdd>_<hhmmss>``
	Returns: 
		dt (datetime): Formatted timestamp including date for calculation of time differences.
	'''
	try:
		dt = datetime.datetime.strptime(fname, f'VNA_%y%m%d_%H%M%S') 
	except ValueError: # 12 PM is not interpreted correctly by the 24 hour clock hour format (https://docs.python.org/3/library/time.html#time.time)
		dt = datetime.datetime.strptime(fname, f'VNA_%y%m%d_%H%M%S') 
	return dt


def read_timestamp_nanovna(fname):
	'''Read timestamp from miniVNA file. 
	Input: 
		fname (str): filename example: ``20241203_16_38_42``.
	Returns: 
		dt (datetime): Formatted timestamp including date for calculation of time differences.
	'''
	dt = datetime.datetime.strptime(fname, f'%Y%m%d_%H_%M_%S') 
	return dt


def read_timestamp_nanovna_api(fname):
	'''Read timestamp from miniVNA file. 
	Input: 
		fname (str): filename example: ``20241203_16_38_42``.
	Returns: 
		dt (datetime): Formatted timestamp including date for calculation of time differences.
	'''
	fname_split = fname.split('_')[-1]
	t_ms = int(fname_split)
	t_s = t_ms/1000 
	return t_s


def read_timestamp_vna(fname):
	'''Read timestamp from VNA file saved with Command Expert Python interface. 
	Input: 
		fname (str): filename example: ``20250821_143154_997``.
	Returns: 
		dt (datetime): Formatted timestamp including date for calculation of time differences.
	'''
	# Split into the datetime part and milliseconds part
	base, ms = fname.rsplit('_', 1)
	# Parse the base part normally
	dt = datetime.datetime.strptime(base, '%Y%m%d_%H%M%S')
	# Add milliseconds as integer (datetime processes microseconds, not milliseconds)
	dt = dt.replace(microsecond=int(ms) * 1000)
	return dt

