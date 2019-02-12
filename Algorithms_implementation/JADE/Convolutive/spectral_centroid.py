import numpy as np
from tqdm import tqdm
from perm import perm


### SPECTRAL SMOOTHNESS OVER TIME ###

def spectral_centroid(Y, t0, t, d, band, bands_centroids) : 

	# Y is the tensor in which the 3rd dimension is related to frequency bands
	# t0 is a strating time
	# t is the time increment
	# d is the depth until which we explore the tree
	# MDCT_freq_size is the number of frequency bands of the MDCT
	# bands_centroids is a storage list for already computed bands centroids
	
	# The idea is to compute spectral centroids including already computed centroids so that our information gets more and more accurate


	# compute the the spectral centroîd for a given time t for each branch of the tree
	N = Y.shape[0]
	MDCT_freq_size = Y.shape[2]
	freq = [1/(MDCT_freq_size)*(band*d + k + 1/2 + MDCT_freq_size/2) for k in range(d)] # center frequency for each band
	PERM = perm(d, N)

	centroids = []

	for b in range(3**d) :
	
		centroid = 0
		samples = []

		for k in range(d) :


			centroid += Y[int(PERM[b, k]), t0 + t, k]*freq[k]
			samples.append(Y[int(PERM[b, k]), t0 + t, k])
	
		centroid = centroid/(np.sum(samples))
		
		for i in range(band) :
			centroid += centroid + bands_centroids[t, i]
		
		if band != 0 :
			centroid = centroid/band
		
		centroids.append(centroid)	

	return centroids