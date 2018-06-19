##########################################################################

# Suite of functions to backpropagate and train a network based on training data

# Using error function E = 1/2 * sum of (differences)^2 for ease

# Example of use are in recipe.py where I show how these tools can be used with
# randomly generated neural networks. The backpropagation seems to reduce all the
# terms of the error function.

# TO DO: How will the step size be chosen?

# Notes on construction and how I kept sane with all the indicies:
#	For multidimensional lists I used a notation where, for example, ABC, is a 3D list.
#	ABC[a][b][c] is an element of the 3D list. And ABCD is a 4D list where ABCD[a] is a
#	3D list with indicies [b][c][d].

##########################################################################

import sigmoid as sig
import forwardProp as fp
import math


def sum_1D_list(A):
	# Returns sum over elements of a list

	s=0
	for element in A:
		s+=element

	return s



def list_times_number(l,n):
	# Assumes list of numbers

	new_list=[]
	for element in l:
		new_list.append(n*element)

	return new_list



def matrix_times_number(AB,n):
	# Returns new matrix where all elements of given m are multiplied by given n

	i_max=len(AB)
	new_matrix=[]

	i=0
	while i<i_max:
		new_matrix.append(list_times_number(AB[i],n))
		i+=1

	return new_matrix



def matrix_from_two_lists(A,B):
	# Returns matrix M where M[a][b]=A[a]*B[b]. This is just the outer product. Order matters: A's index will
	# be the first index. B's is second

	a_max=len(A)-1
	b_max=len(B)-1
	M=[]
	a=0
	while a<=a_max:
		b=0
		temp_list=[]
		while b<=b_max:
			temp_list.append(A[a]*B[b])
			b+=1
		M.append(temp_list)
		a+=1

	return M



def list_Hadamard_list(A,B):
	# Returns Hadamard product of A and B which is a list.

	new_list=[]

	if abs(len(A)-len(B))>1:
		print "!!!!!!!!!!!!!!!!*****************!!!!!!!!!!!!!!!!!!! Bad list_Hadamard_list. Difference in max index values > 1"
	else:
		a_max=len(A)-1
		b_max=len(B)-1
		if b_max<a_max:
			a_max=b_max
		a=0
		while a <= a_max:
			new_list.append(A[a]*B[a])
			a+=1

	return new_list



def list_Hadamard_matrix(A,BC):
	# Given list A[a] and matrix BC[b][c]:
	# 	Returns a matrix M where M[b][c]=A[b]*BC[b][c]
	# I will assume that len(A)==len(B)

	a_max=len(A)-1
	b_max=len(BC)-1
	M=[]
	
	if b_max<a_max:
		a_max=b_max

	a=0
	while a<=a_max:
		M.append(list_times_number(BC[a],A[a]))
		a+=1

	return M



def matrix_Hadamard_list(AB,C):
	# Returns a matrix M where M[a][b]=AB[a][b]*C[b]
	# I will assume that len(A)==len(B)
	
	b_max=len(AB[0])-1
	c_max=len(C)-1
	M=[]

	if c_max<b_max:
		b_max=c_max

	b=0
	while b<=b_max:
		M.append(list_Hadamard_list(AB[b],C))
		b+=1

	return M



def matrix_Hadamard_matrix(AB,CD):
	# Given AB = W[l] for some layer l with indicies a,b and a matrix CD with indicies c,d
	# Returns a new list of matrices M with indicies a,c,d. M[a][c][d] = AB[a][c] * CD[c][d]

	# Special circumstance here is AB[a][b] has a biased index b compared with CD[c][d] index c. b_max = c_max + 1.
	# Fix here is to use the index c and its c_max to do the Hadamard product and the bias term at b_max is ignored.

	a_max=len(AB)-1

	M=[]
	a=0
	while a<=a_max:
		M.append(list_Hadamard_matrix(AB[a],CD))
		a+=1

	return M



def sum_lists(A,B):
	# Given two lists, return element by element sum which is also a list

	new_list=[]
	a_max=len(A)-1
	a=0
	while a<=a_max:
		new_list.append(A[a]+B[a])
		a+=1

	return new_list



def sum_2Dmatrices(AB,CD):
	# Given two 2D matrices, return new matrix MN where MN[m][n]=AB[m][n]+CD[m][n]
	# I assume that the matrices are the same sizes

	a_max=len(AB)-1
	a=0
	MN=[]
	while a<=a_max:
		MN.append(sum_lists(AB[a],CD[a]))
		a+=1

	return MN



def flatten(ABC):
	# Given ABC which must be a * nice * 3D matrix M[a][b][c], returns a matrix DE where DE[b][c] = SUM OVER a of ABC[a][b][c]
	# I assume ABC has 3 dimensions of numbers and * nice * means all the ABC[a] have the same dimensions.

	zeroes_list=[]
	c_max=len(ABC[0][0])-1
	c=0
	while c<=c_max:
		zeroes_list.append(0)
		c+=1
	DE=[]
	b_max=len(ABC[0])-1
	b=0
	while b<=b_max:
		DE.append(zeroes_list)
		b+=1

	a_max=len(ABC)-1
	a=0
	while a<=a_max:
		DE=sum_2Dmatrices(DE,ABC[a])
		a+=1
	return DE
			


def flatten_2nd_index(ABCD):
	# Given 4D matrix ABCD[a][b][c][d], builds 3D matrix EFG[a][c][d] where EFG[a] is flatten of ABCD[a]

	EFG=[]
	a_max=len(ABCD)-1
	a=0
	while a<=a_max:
		EFG.append(flatten(ABCD[a]))
		a+=1

	return EFG



def sum_sublists(MN):
	# Given a 2D matrix MN, returns a new list A of length m_max where each element is the
	# position-wise sum of elements. For example: sum_sublists([[1,2],[2,3],[3,4]]) returns [6,9]

	m_max=len(MN)-1 		# list length
	n_max=len(MN[0])-1	# sublist length

	accumulant=[]
	n=0
	while n<=n_max:
		m=0
		sub_sum=0
		while m<=m_max:
			sub_sum+=MN[m][n]
			m+=1
		accumulant.append(sub_sum)
		n+=1

	return accumulant



# I don't think I need this method but it was fun to make
# def transpose(matrix):
# 	# Assumes sublists all have the same length

# 	new_matrix=[]
# 	i_max=len(matrix)
# 	j_max=len(matrix[0])
# 	j=0
# 	while j<j_max:
# 		i=0
# 		vec=[]
# 		while i<i_max:
# 			vec.append(matrix[i][j])
# 			i+=1
# 		new_matrix.append(vec)
# 		j+=1

# 	return new_matrix



def biased_output(l,outs):
	
	response=[]
	for element in outs[l]:
		response.append(element)
	response.append(-1.0)

	return response



def error_difference_list(outs,A):
	# Given outs and answers A, returns a list of error differences

	l_max=len(outs)-1
	
	return sum_lists(outs[l_max],list_times_number(A,-1.0))



def dsig_list(l,outs):
	# Given layer index l and outs matrix, return a list which is Hadamard(outs[l],1-outs[l])

	# Create the 1 list:
	i_max=len(outs[l])-1
	i=0
	one_list=[]
	while i<=i_max:
		one_list.append(1.0)
		i+=1

	return list_Hadamard_list(outs[l],sum_lists(one_list,list_times_number(outs[l],-1.0)))



def list_Hadamard_3Dmatrix(A,BCD):
	# Takes list A[a] and 3D matrix BCD[b][c][d] and returns the Hadamard product M[a][c][d]=A[a]*BCD[a][c][d]

	a_max=len(A)-1
	b_max=len(BCD)-1
	if b_max<a_max:
		a_max=b_max
	M=[]
	a=0
	while a<=a_max:
		M.append(matrix_times_number(BCD[a],A[a]))
		a+=1

	return M



def matrix_Hadamard_3Dmatrix(AB,CDE):
	# Builds 4D matrix M where element M[a][b][d][e] = AB[a][b]*CDE[b][d][e]

	a_max=len(AB)-1
	M=[]
	a=0
	while a<=a_max:				# Indexes M
		M.append(list_Hadamard_3Dmatrix(AB[a],CDE))
		a+=1
	
	return M



def recurse(d,W,outs,l,l_max): 
	# Returns list of matrices M[i][m][n]. i is node index from layer l_max, m is node index from layer l_max-1, n is node
	# index from layer l_max-2.

	if d==0:							
		root = matrix_Hadamard_matrix(W[l+1],matrix_from_two_lists(dsig_list(l,outs),biased_output(l-1,outs)))
		response = root

	else:
		stepstone=list_Hadamard_3Dmatrix(dsig_list(l+d,outs),recurse(d-1,W,outs,l,l_max))
		response=flatten_2nd_index(matrix_Hadamard_3Dmatrix(W[l+d+1],stepstone))
	
	return response



def dEdW(A,W,outs,l):

	l_max=len(outs)-1
	d=l_max-l
	preamble=list_Hadamard_list(error_difference_list(outs,A),dsig_list(l_max,outs))
	
	if d == 0:
		response = matrix_from_two_lists(preamble,biased_output(l_max-1,outs))
	else:
		response = flatten(list_Hadamard_3Dmatrix(preamble,recurse(d-1,W,outs,l,l_max)))

	return response



def dEdW_table(A,W,outs):
	# Builds a table with same shape as W.

	l_max=len(outs)-1
	table=[]
	table.append(0)			# W's 0th element is a zero
	l=1
	while l<=l_max:
		table.append(dEdW(A,W,outs,l))
		l+=1

	return table



def Pythagorean_size_3Dmatrix(ABC):
	# Given 3D matrix ABC[A][B][C], returns a pythagorean size = sqare root of sum of squares of all of its elements.
	# To this end, we have to ratchet through all combos of indexes carefully because all of the ABC[a] are not the same size.

	# Also, since index 'a' is a layer index, its domain is [1,l_max], not [0,l_max].

	# *********  CAUTION!! can return ZERO. 

	a_max=len(ABC)-1
	sum_of_squares=0
	a=1
	while a<=a_max:
		b_max=len(ABC[a])-1
		b=0
		while b<=b_max:
			c_max=len(ABC[a][b])-1
			c=0
			while c<=c_max:
				sum_of_squares+=math.pow(ABC[a][b][c],2)
				c+=1
			b+=1
		a+=1

	return math.sqrt(sum_of_squares)



def update_W(A,W,outs,step):
	# Given answers list A, current weights W, and outputs of all nodes of network outs, updates
	# W to reduce the error function: W[l][m][n] = W[l][m][n]-step * dEdW[l][m][n] / Pyth_size(dEdW)

	table=dEdW_table(A,W,outs)
	norm=Pythagorean_size_3Dmatrix(table)

	new_W=[]
	if norm != 0.0:
		shift=-step/norm
		new_W.append(0)
		l_max=len(outs)-1
		l=1
		while l<=l_max:
			temp_matrix=[]
			m_max=len(W[l])-1
			m=0
			while m<=m_max:
				temp_list=[]
				n_max=len(W[l][m])-1
				n=0
				while n<=n_max:
					temp_list.append(W[l][m][n]+shift*table[l][m][n])
					n+=1
				temp_matrix.append(temp_list)
				m+=1
			new_W.append(temp_matrix)
			l+=1
	else:
		new_W=W

	return new_W
