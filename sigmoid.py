##########################################################################

# sigmoid returns sigmoid evaluation for sig(z)=1/(1+e^(-x))
# dsig uses sigmoid to calculate derivative of sigmoid which, in this case, is dsig(z)=sig(z)*(1-sig(z))

# Note: does not make any checks. Meant to be used internally.

# As suggested on web at https://takinginitiative.wordpress.com/
# 2008/04/03/basic-neural-network-tutorial-theory/, if sigmoid(z) > .9, just call it 1;
# if sigmoid(z) < .1, just call it zero. These cutoffs are at domain values z = +/- 2.2

##########################################################################

import numpy

def sig(z):
	if z > 2.2:
		val = 1.0
	elif z < -2.2:
		val = 0.0
	else:
		val = 1.0/(1+numpy.exp(-1.0*z))

	return val

def dsig(z):
	sigma = sig(z)

	return sigma*(1.0-sigma)