##########################################################################

# Don't know how to choose the domain for these. Starting with uniform prob density from interval [0,1]

# Notes on indicies of W:
# 	Returns a 3D list W[l][i][j] where l is layer index, i is node index in layer l, j is node index in layer l-1.
# 	Topology vector will be [s0, s1, s2, ... , sL] where s0 is number of input nodes of input layer, s1 is number of nodes of
# 	first hidden layer, etcetera. Node labels will come from [0,sl-1]. That is, node counting starts at zero. Also, we
# 	do not need weights for input nodes so W[0] will just be an unused zero. But W[not zero] will be a matrix with indicies i,j.

# 	If the topology list is [s_0,s_1,...,s_(l_max-1)], then the biases are the elements W[l][m][s_(l-1)].
# 	Also, for example, W[l][i][0] is the weight for output of the first node of layer l-1.

##########################################################################

import random as r

def random_weights(topo):

	l_max=len(topo)-1 							# Number of hidden layers is L-1
	W=[] 									# Start creation of weights matrix
	W.append(0) 							# The unused zero
	l=1 									# layer index counter
	while l <= l_max:
		m=0 								# counter for nodes
		m_max=topo[l]-1
		temp_matrix=[]
		while m <= m_max: 					# t[l] is current layer's size
			n=0
			n_max=topo[l-1] 					# Extra bias term so no subtraction by 1
			temp_list=[] 
			while n <= n_max:
				temp_list.append(r.random()) 	# building weights and a bias for outputs of previous layer
				n+=1
			temp_matrix.append(temp_list)	
			m+=1
		W.append(temp_matrix)	
		l+=1

	return W
