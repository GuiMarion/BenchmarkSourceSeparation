import wave
import soundfile as sf
import numpy as np
import scipy
from Jade import jadeR

from tqdm import tqdm

################################################
# We consider 3 stereo observations of 3 sources
# We get a 3 mono audio files
################################################

Audiofiles_Path = "/Users/Schenkel/Documents/COURS/ATIAM/S4/PAM/JADE/Audio"
Sensors_names = ["Gtr","Cello","Clar"]
N = len(Sensors_names)

### WAY TO DETERMINE THE NUMBER OF FRAMES ###

data = wave.open(Audiofiles_Path + "/" + "Verb" + "_" + Sensors_names[0] + ".wav")
T = data.getnframes()
SR = data.getframerate()

### CREATE X MATRIX OF OBSERVATIONS ###

X = np.zeros((2*N,T))

for i in tqdm(range(N)) :

	data = sf.read(Audiofiles_Path + "/" + "Verb" + "_" + Sensors_names[i] + ".wav")
	data = np.asarray(data).reshape(-1)[0]
	X[2*i,:] = data[:,0]
	X[2*i+1,:] = data[:,1]


### COMPUTING JADE ###

B = jadeR(X,N)
Y = B*X
Y = [Y[i]/np.max(Y[i]) for i in range(N)]	# normalization for each source

sources = np.arange(N)
sources = [str(sources[i]) for i in range(N)]

### CREATING SEPARATED SOURCES .WAV ###



for i in range(N) :

	sf.write(Audiofiles_Path + "/" + "SEPARATION" + "/" + "Jade" + "/" + "Verb" + "_" + "source_" + sources[i] + ".wav", np.array(Y[i]).reshape(len(X[0])), SR)
	