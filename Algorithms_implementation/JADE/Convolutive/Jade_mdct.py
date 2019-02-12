import wave
import soundfile as sf
import numpy as np
import scipy
from Jade import jadeR
from STFT import computeSTFT
from mdct_compute import mdct, imdct
from spectral_centroid import spectral_centroid
from Sort import sorted_sources
from perm import perm
from tqdm import tqdm

##################################################
# We consider 2*N stereo observations of N sources
# We get N mono audio files
##################################################

Audiofiles_Path = "/Users/Schenkel/Documents/COURS/ATIAM/S4/PAM/JADE/Audio"
Sensors_names = ["Gtr","Cello","Clar"]
N = len(Sensors_names)

### WINDOWING ###
# Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format

window_length = 512
alpha_value = 5
window_function = np.kaiser(int(window_length/2)+1, alpha_value*np.pi)
window_function2 = np.cumsum(window_function[0:int(window_length/2)])
window_function = np.sqrt(np.concatenate((window_function2, window_function2[int(window_length/2)::-1]))/ np.sum(window_function))

### GET SAMPLE RATE ###

data = wave.open(Audiofiles_Path + "/" + Sensors_names[0] + ".wav")
T = data.getnframes()
SR = data.getframerate()

data = sf.read(Audiofiles_Path + "/" + Sensors_names[0] + ".wav")
data = np.asarray(data).reshape(-1)[0]

### GET MDCT SIZE ###

MDCTl = mdct(data[:,0], window_function)
MDCT_freq_size, MDCT_time_size  = MDCTl.shape

### CREATE X TENSOR OF TIME-FREQUENCY OBSERVATIONS ###

X = np.zeros((2*N, MDCT_time_size, MDCT_freq_size))

for i in tqdm(range(N)) :

	data = sf.read(Audiofiles_Path + "/" + Sensors_names[i] + ".wav")
	data = np.asarray(data).reshape(-1)[0]

	MDCTl = mdct(data[:,0], window_function)
	MDCTr = mdct(data[:,1], window_function)

	for j in range(MDCT_time_size) :

		X[2*i, j, :] = MDCTl[:, j]
		X[2*i + 1, j, :] = MDCTr[:, j]

# We obtain a tensor X with source x time x frequency

### COMPUTING JADE FOR EACH FREQUENCY BIN ###

# We apply JADE algorithm to every frequency band of the signal 

Y = np.zeros((N, MDCT_time_size, MDCT_freq_size))	

for i in range(MDCT_freq_size) : 

	B = jadeR(X[:, :, i], N)
	Y[:, :, i] = B*X[:, :, i]

### SPECTRAL SMOOTHNESS OVER TIME ###

# We use sorted_sources function to perform a sorting of the sources along the frequency bins

d = 3	# d is the depth until which we explore the tree
t0 = 2000	# time from which we start observing the centroïd
l = 20	# duration on which we observe the evolution of the centroid

Y_sorted = sorted_sources(Y, d, t0, l)


### CREATING SEPARATED SOURCES .WAV ###

IMDCT_size = imdct(Y_sorted[0, :, :].T, window_function).shape[0]
sources = np.zeros((N, IMDCT_size))	

sources_names = np.arange(N)
sources_names = [str(sources_names[i]) for i in range(N)]

for i in range(N) :

	sources[i, :] = imdct(Y_sorted[i, :, :].T, window_function)
	sf.write(Audiofiles_Path + "/" + "SEPARATION" + "/" + "Jade_mdct" + "/" + "source_" + sources_names[i] + ".wav", np.array(sources[i]).reshape(len(sources[i])), SR)






