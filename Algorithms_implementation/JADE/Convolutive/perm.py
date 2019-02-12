import numpy as np

### Compute a matrix in which lines are possible branches of depth d with N sons nodes ###

def perm(d, N) :

	perm = np.zeros((N**d, d))

	for i in range(3**(d-1)) :
		
		for n in range(N) : 

			perm[3*i + n, d-1] = n
		
	for j in range(1, d) :

		for i in range(N**(d-j-1)) :
			for n in range(N) :

				perm[N**(j+1)*i + N**j*n : N**(j+1)*i + N**j*(n+1), d-j-1] = [n]*(N**j)

	return perm 

