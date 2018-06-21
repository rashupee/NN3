################################################

# A training script for rectangles training data located relative to where this file is kept at:
# ./rectangles/rectangles_train.amat

# I think this .amat file is simply text with spaces and new lines. Quoted from where I got the data:
# http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/RectanglesData 

# "Each archive contains two files -- a training (and validation) set and a test set. We
# used the last 2000 examples of the training sets as validation sets for rectangles-images
# and 200 for rectangles. In the case of SVMs, retrained the models with the entire set after
# choosing the optimal parameters on these validation sets. Data is stored at one example per
# row, the features being space-separated. There are 784 features per example (=28*28 images),
# corresponding to the first 784 columns of each row. The last column is the label,
# which is 1 or 0."

###############################################

# Make sure I can open the file and then build a list of dictionaries: rectangles_data

# rectangles_data = [dict1,dict2,...] where dict1={'feature_data':[list of 784],'answer':float}

data_list=[]

f = open('rectangles/rectangles_train.amat', 'r')

for line in f:
	temp_dict={}
	temp_list=line.split()
	for element in temp_list:
		element=float(element)
	temp_answer=temp_list.pop()
	temp_dict={'feature_data':temp_list,'answer':temp_answer}
	data_list.append(temp_dict)

f.close()
# print data_list[0]

print "Length of features_list is ",len(data_list[0]['feature_data'])
print "Number of training data elements is ",len(data_list)


