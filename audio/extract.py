import soundfile as sf
from glob import glob
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def fadeOutFunction(x):
	#return 1 - np.exp(x/15000)
	#return 1/np.exp(x/15000)
	#return -x/96000 + 1

	return np.log((1-np.exp(1))/96000*x + np.exp(1))

def fadeOut(X):
	hey = []
	for i in range(len(X)):
		if i < 95000:
			X[i][0] *= fadeOutFunction(i)
			X[i][1] *= fadeOutFunction(i)
		else:
			X[i][0] *= 0
			X[i][1] *= 0

		#hey.append(X[i][0])

	#plt.plot(hey)
	#plt.show()
	return X

def fadeInFunction(x):
	#return 1 - np.exp(x/15000)
	#return 1/np.exp(x/15000)
	#return -x/96000 + 1

	return 2*(np.log((np.exp(1) -1)/720*x + np.exp(1)) -1 )

def fadeIn(X):
	hey = []
	for i in range(len(X)):
		X[i][0] *= fadeInFunction(i)
		X[i][1] *= fadeInFunction(i)

		#hey.append(fadeInFunction(X[i][0]))

	#plt.plot(hey)
	#plt.show()
	return X

def extract(path, start, end):
	data, samplerate = sf.read(path)

	# extraction
	data =  data[int(start*samplerate):int(end*samplerate)]

	data[:720] = fadeIn(data[:720])

	# fade out
	data[-96000:] = fadeOut(data[-96000:])

	# saving
	sf.write(path, data, samplerate)

def extractFolder(folder, start, end):
	for file in tqdm(glob(folder+"*")):
		extract(file, start, end)


if __name__ == "__main__":

	if len(sys.argv) == 2 :
		extractFolder(sys.argv[1], 45, 58.5)

	else:
		print("Usage: Python3 extract.py <folder>")  

