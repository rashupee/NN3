# Suite of functions associated with taking in an input and forward propagating with weights to a network answer.

import sigmoid



def node_output(in_put,W_l_i):
	# Given input and W[l][i], returns output of node i in layer l

	j_max = len(in_put)-1
	j = 0					# counter over node outputs from previous layer
	z = 0					# initialize the argument for sigmoid
	while j <= j_max:
		z += in_put[j] * W_l_i[j]
		j += 1

	return sigmoid.sig(z-W_l_i[j]) # Subtract the bias term



def output_from_layer(in_put,W_l):
	# Given input and W[l], returns output of nodes of layer l

	layer_output=[]
	i_max=len(W_l)-1
	i=0
	while i <= i_max:
		layer_output.append(node_output(in_put,W_l[i]))
		i+=1

	return layer_output



def outputs_from_network(in_put,W):
	# Given input to network and weights, returns outputs of all layers (l=0,1,2,...,L-1) as matrix
	# l=0 is the inputs layer and outputs[0] is just inputs.
	# outputs[L-1] is network's guess based on in_put

	outputs=[]
	outputs.append(in_put)

	l=1
	l_max=len(W)-1

	while l <= l_max:
		outputs.append(output_from_layer(outputs[l-1],W[l]))
		l+=1

	return outputs
