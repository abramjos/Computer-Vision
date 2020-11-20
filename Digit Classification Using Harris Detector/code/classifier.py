import os
import contextlib
from harris_detector import *
from tqdm import tqdm
from glob import glob
import pandas as pd 
import heapq

#read CSV data and arrange the dataset
def read_data(data_val,mask,size = 50,thresh=8):
    train = pd.read_csv(data_val)
    data_vec = [[] for i in range(10)]
    for ix,i in tqdm(enumerate(train.values)):
        if abs(i[2])> thresh:
            continue
        image_id = i[1] 
        if len(data_vec[image_id]) >= size:
            if np.array(data_vec).shape == (10, 50, 20):
                print('each class has 50 features') 
                break
            else:
                continue
        image_file = data_val.replace(data_val.split('/')[-1],'{}/{}'.format(image_id,i[0]))
        image = cv2.imread(image_file,0)

        #To supress the priniting
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            R,_,_,_,_ = harris_detector_conv(image,mask,gaus=False)

        #finding the largest and smallest 10 features from R 
        R_flatten = R.flatten()
        feature_low_10 = heapq.nsmallest(10,R_flatten)
        feature_high_10 = heapq.nlargest(10,R_flatten)
        feature_20 = np.array((feature_low_10 + feature_high_10 ))
        data_vec[image_id].append(np.array(feature_20).reshape((20,1)))

    data = [np.array(i) for i in data_vec]
    return(np.array(data))

# creating the LDF response 
def LDF_resp(feature_20,cov_matrix,mean_vec):
    g_resp = np.zeros((no_class))

    features = feature_20
    cov_inv = np.linalg.inv(cov_matrix)
    # import ipdb;ipdb.set_trace()    

    for i in range(no_class):
        hi = np.dot(cov_inv,mean_vec[i])
        out = np.dot(np.transpose(features),hi) - 1/2.0*np.dot(np.transpose(mean_vec[i]),hi)
        g_resp[i]=out[0][0]

    resp = np.argmax(g_resp)

    return(resp)

#calculating performance in the dataset
def obtain_res(feature_vec,cov_matrix,mean_vec):
    no_class = feature_vec.shape[0]
    cm = np.zeros((no_class,no_class)) 
    for ix,i in enumerate(feature_vec):
        for j in i:
            o = LDF_resp(j.reshape(20,1),cov_matrix,mean_vec)
            cm[ix,o]+=1
    return(cm)


# train_image = glob('../data/mnist/train/*.jpg')
# test_image = glob('../data/mnist/test/*.jpg')
# feature_vec = create_feature_vec(train_image)
# test_vec = create_feature_vec(test_image)


#read data form the train and test dataset
data = ['../data/DigitDataset/digitTrain.csv','../data/DigitDataset/digitTest.csv']
data_val = data[0]

mask=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

#getting the feature vector from the training and  test set. 
feature_vec = read_data(data[0],mask)
test_vec = read_data(data[1],mask)


no_feature = len(feature_vec[0][0])  
no_class = len(feature_vec) #give any class number to get distribution from 0-i class.
# no_class = 3
no_sample = len(feature_vec[0])

#finding the mean of the classes
mean_vec = np.zeros([no_class,no_feature,1])
for i in range(no_class):
    mean_vec[i] = np.average(feature_vec[i],axis=0) 
mean_vec = np.array(mean_vec)

#overall mean for all classes
overall_mean = np.average(mean_vec,axis=0)

#finding the covariance matrix
cov_matrix = np.zeros((no_feature,no_feature))
for i in range(no_class):
    for j in range(no_sample):
        vec = np.array(feature_vec[i][j] - overall_mean)
        cov = np.dot(vec,np.transpose(vec))
        cov_matrix += cov

cov_matrix = cov_matrix/(no_sample*no_class)

plt.subplot(212)
ax1=plt.subplot(211)
ax2=plt.subplot(212)

#plotting the mean vec
for i in range(no_class):
    ax1.plot(mean_vec[i], alpha=0.5, label = 'class-{}'.format(no_class))
ax1.legend(loc=2)
ax1.set_title('DigitDataset mean')
#plotting the covariance matrix
ax2.imshow(cov_matrix)
ax2.set_title('Covariance overall')
#plt.title('Plot of Co-variance and Mean values of features')
plt.savefig('../data/mean_DigitDataset.png')
plt.show()


train_res = obtain_res(feature_vec,cov_matrix,mean_vec)
test_res = obtain_res(test_vec,cov_matrix,mean_vec)

train_error_rate = np.sum(train_res*np.eye(10))/500
test_error_rate = np.sum(test_res*np.eye(10))/500

print('Error Rate\nTraining error rate - {}\nTesting error rate - {}'.format(train_error_rate,test_error_rate))