import numpy as np

from spectral_centroid import spectral_centroid
from perm import perm
from tqdm import tqdm

# The idea is to compute an average of the differences between centroids at t and t+1 on a short time l
# by frequency bands (groups of bins of depth d) for the whole tree

def sorted_sources(Y, d, t0, l) :

	# Y is the tensor of sources spectrograms to be sorted
	# d is the depth until which we explore the tree
	# temps (sample) à partir du quel on observe l'évolution du centroïd
	# durée sur laquelle on observe l'évolution du centroïd

	N, MDCT_time_size, MDCT_freq_size = Y.shape
	

	Y_sorted = np.zeros((N, MDCT_time_size, MDCT_freq_size))

	moy = np.zeros(3**d)

	PERM = perm(d, N)	# compute the matrix of permutations

	bands_centroids = np.zeros((l,MDCT_freq_size//d))	# storage for already computed bands centroids

	for band in tqdm(range(MDCT_freq_size//d)) :
	
		ct1 = []
		ct = np.zeros((l, 3**d))	# temporary storage for centroids

		for t in tqdm(range(l-1)) :

			ct1 = spectral_centroid(Y, t0, t, d, band, bands_centroids)
			ct2 = spectral_centroid(Y, t0, t + 1, d, band, bands_centroids)
			ct1 = np.asarray(ct1)
			ct[t, :] = ct1
		

			delta = [ct1[i] - ct2[i] for i in range(3**d)]
			delta = [abs(delta[i]) for i in range(3**d)]
	
			for i in range(3**d) : 
				moy[i] += delta[i]/l

		matching_branch_index = np.argmin(moy)
		print(matching_branch_index)
		bands_centroids[:, band] = ct[:, matching_branch_index]
	
		#print(perm[matching_branch_index, 3])

		for k in range(d) :

			Y_sorted[: , :, band*d + k] = Y[int(PERM[matching_branch_index, k]), :, band*d + k]

	return Y_sorted