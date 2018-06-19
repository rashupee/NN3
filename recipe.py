##########################################################################

# This script randomly picks a network topology, builds a weights table for that topology,
# and generates a random list of inputs and answers that fits the topology.

# Then it iterates backpropagation on the single random inputs and answer combination to
# show how the errors are changing.

# I've noticed that the errors are always decreasing for multiple random topologies-
# so, I think this is roughly working.

# Still have to come up with a way to determine the step size argument for theupdate_W in
# function in backProp.py


##########################################################################

import backProp as bp
import forwardProp as fp
import randomWeights as rw
import random as r


## Random topo for a random network
topo=[]
l_max=r.randint(1,10)
l=0
while l<=l_max:
	topo.append(r.randint(1,10))
	l+=1



## Random weights table
W=rw.random_weights(topo)



## Random inputs
in_puts=[]
n_max=topo[0]-1
n=0
while n<=n_max:
	in_puts.append(r.random())
	n+=1



## Random answers A
A=[]
n_max=topo[len(topo)-1]-1
n=0
while n<=n_max:
	A.append(r.random())
	n+=1



## Check to see if the error difference list always shows a decrease in error for all the answers.
## This is noted by watching all the columns in the output of the following script decrease as it iterates.

count=0
count_max=500
while count<=count_max:
	outs=fp.outputs_from_network(in_puts,W)		# Produce a network guess
	print bp.error_difference_list(outs,A)		# Show the list of differences between network guess and answer
	W=bp.update_W(A,W,outs,.01)					# Update the weights with backpropagation
	count+=1
