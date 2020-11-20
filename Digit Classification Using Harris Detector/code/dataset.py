import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


mndata = MNIST('./samples/')
images, labels = mndata.load_testing()

lst_data= [0]*10
for ix,i in enumerate(labels):
	if lst_data[i] <= 99:
		image = np.reshape(np.array(images[ix]),(28,28))
		image = image.astype(np.uint8)
		
		lst_data[i]+=1

		if lst_data[i] <= 50:
			cv2.imwrite('mnist/train/im_{}-{}.jpg'.format(i,ix), image)
		else:
			cv2.imwrite('mnist/test/im_{}-{}.jpg'.format(i,ix), image)
	else:
		continue


