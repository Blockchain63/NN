import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import functions as F

def main():

	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	m_train = train_set_x_orig.shape[0] ##### m=n (DUNNO WHY BUT IT IS) 209
	m_test = test_set_x_orig.shape[0]   ##### same for this one 50
	num_px = train_set_x_orig.shape[2]  ##### 64 (cuz image is 64x64)

	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T ### at this point we need to turn X to make it vertical
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T    ### and make it from (209,64,64,3) into (12288,209)

	train_set_x = train_set_x_flatten/255. #### now every pixel is nomore 0-255, it's 0-1 (make's a little sense here, but useful when input is various
	test_set_x = test_set_x_flatten/255.



	d = F.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)


	F.analysis([0.01, 0.001], train_set_x, train_set_y, test_set_x, test_set_y)
	##################

	#############EXAMPLE

	my_image = "123.jpg"   # change this to the name of your image file 
	## END CODE HERE ##

	# We preprocess the image to fit your algorithm.
	fname = "images/" + my_image
	image = np.array(ndimage.imread(fname, flatten=False))
	my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
	my_predicted_image = F.predict(d["w"], d["b"], my_image)

	plt.imshow(image)
	print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

if __name__ == "__main__":
  main()
