'''

Assignment 4 for Computer Vision- CAP 5415.
Question 1: Harris feature classifier using Fisher space
Abraham Jose
11-14-2019
abraham@knights.ucf.edu 

'''
from math import floor,sqrt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import heapq
from math import floor
import cv2
import os
import pickle,contextlib
from tqdm import tqdm

class harris_detector:
    def __init__(self,gradient_mask=np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), gaussian = False, window_size = 3):
        
        self.gradient = gradient_mask

        if gaussian == True:
            self.window = gkern(window_size,3)
        else:
            self.window = np.ones((window_size,window_size))


    #funtion to create a gaussian kernel
    def gkern(len=3, nsig=3):

        #creating interval based on sigma value
        interval = (2*nsig+1.)/(len)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., len+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel


    #function for convolution
    def convolve(self,image,mask,pad=True):

        #getting the mask shape and flipping the mask
        mask_shape,_=mask.shape
        mask_flipped=np.flip(np.flip(mask,1),0)

        #intilialize resultant matrix with n+m-1 size; n-image size, m-mask size
        result=np.zeros(np.array(image.shape)+np.array((mask_shape-1,mask_shape-1)))

        #creating a image padding for convolution
        image_padded=result.copy()
        offset=floor(mask_shape/2)
        image_padded[offset:image_padded.shape[0]-offset,offset:image_padded.shape[1]-offset]=image


        for i in range(offset,result.shape[0]-offset):
            for j in range(offset,result.shape[1]-offset):
                #getting the result of matrix multiplication and taking the sum
                result[i,j]=np.sum(mask_flipped*image_padded[i-offset:i-offset+mask_shape,j-offset:j-offset+mask_shape])
                # import ipdb;ipdb.set_trace()
        if pad==False:
            return(result[offset:image_padded.shape[0]-offset,offset:image_padded.shape[1]-offset])
        else:
            return(result)

    #function for edge detector
    def edge_detector(self,image,pad=False):
        
        #creating mask for x and y
        mask_x=self.gradient
        mask_y=np.flip(np.rot90(self.gradient),0)

        #getting gradients in x nad y directions
        grad_x=self.convolve(image,mask=mask_x,pad=pad)
        grad_y=self.convolve(image,mask=mask_y,pad=pad)

        grad_x=grad_x/grad_x.max()*255.0
        grad_y=grad_y/grad_y.max()*255.0

        return(grad_x,grad_y)



    def harris_detector_conv(self,image_data,k=0.05):

        if len(image_data.shape)==3:
            image=cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
        else:
            image=image_data
    #finding xa nad y gradient
        I_x,I_y = self.edge_detector(image=image)
        mask_shape = self.gradient.shape[0]
        #finding second order gradients
        I_xx=I_x**2
        I_xy=I_x*I_y
        I_yy=I_y**2

        print('Using the window: \n',self.window,'\n')

        sum_xx=self.convolve(mask=self.window,image=I_xx,pad=False)
        sum_yy=self.convolve(mask=self.window,image=I_yy,pad=False)
        sum_xy=self.convolve(mask=self.window,image=I_xy,pad=False)

        #Find the determinant and trace to find corner response using equation L1+L2 and L1*L2
        det=(sum_xx*sum_yy)-(sum_xy**2)
        trace=sum_xx-sum_yy
        R=det-k*(trace**2)

        # vec_values=[]
        # vec_val=[]
        #import ipdb;ipdb.set_trace()

        # for ix,i in enumerate(det):
        #     for iy,j in enumerate(i):
        #         hessian = np.array([[sum_xx[ix,iy],sum_xy[ix,iy]],[sum_xy[ix,iy], sum_yy[ix,iy]]])
        #         l1,l2 = np.linalg.eigvals(hessian) 
        #         vec_values.append(np.array([l1,l2]))
        #         vec_val.append(np.array([l1,l2,R[ix,iy]]))
        # return(R,np.array(vec_values))
        return(R)

    def feature(self,image_data):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            R = self.harris_detector_conv(image_data)
        R_flatten = R.flatten()
        feature_low_10 = heapq.nsmallest(10,R_flatten)
        feature_high_10 = heapq.nlargest(10,R_flatten)
        feature_20 = np.array((feature_low_10 + feature_high_10 ))
        data_vec = np.array(feature_20).reshape((20,1))

        return(data_vec)

    def create_features(self,dataset,no_classes=10):
        features = [[] for i in range(no_classes)] 
        # import ipdb;ipdb.set_trace()
        for idx,class_wise_image in tqdm(enumerate(dataset),total = len(dataset)):
            for image in class_wise_image:
                #finding harris feature vector
                feature_detected = feature_detector.feature(image)
                features[idx].append(feature_detected)
        
        features = [np.array(feature_class_wise) for feature_class_wise in features]        

        return(features)


class fisher:
    def __init__(self):
        self.train = False

    def _cov(self,val,val_mean):
        #finding the covariance matrix
        cov_matrix = []

        for value in val:
            # import ipdb;ipdb.set_trace()
            vec = value - val_mean
            cov = np.dot(vec,np.transpose(vec))
            cov_matrix.append(cov)

        covariance = np.average(np.array(cov_matrix),axis=0)

        return(covariance)


    def classify(self,val,eigen_vector,fisher_cov_inv,fisher_mean):
        classes = ['corner','edge','edge','flat']
        val = np.array(val).reshape((2,1))
        mahalnobis_dist = np.zeros(4)
        feature = np.dot(np.transpose(eigen_vector),val)
        for i,(mean,cov_inv) in enumerate(zip(fisher_mean,fisher_cov_inv)):
            vec = feature-mean
            mahalnobis_dist[i] = np.dot(np.dot(np.transpose(vec),cov_inv),vec)
        feature_class=np.argmin(mahalnobis_dist)

        return(classes[feature_class],mahalnobis_dist)


    def _fisher_space(self,val,eigen_vector):
        fisher_space=[]
        for i in val:
            fisher_space.append(np.dot(np.transpose(eigen_vector),np.transpose(np.array([i]))))
        return(np.array(fisher_space))


    def get_fisher(self,data):
        self.no_classes = int(len(data))

        self.class_mean = np.zeros((self.no_classes,)+data[0][0].shape)
        for idx,class_data in enumerate(data):
            self.class_mean[idx] = np.average(class_data,axis=0)

        self.overall_mean = np.average(np.array(self.class_mean),axis=0)
        
        self.class_cov = np.zeros((self.no_classes,)+(data[0][0].shape[0],data[0][0].shape[0]))
        for idx,class_wise in enumerate(data):
            self.class_cov[idx] = self._cov(class_wise,self.class_mean[idx])

    
        B = np.zeros_like(self.class_cov[0]).astype('float64')
        A = np.zeros_like(self.class_cov[0]).astype('float64')

        for mean,cov in zip(self.class_mean,self.class_cov):
            vec = np.array(mean - self.overall_mean)
            B += np.dot(vec,np.transpose(vec))
            A += cov
        A_inv = np.linalg.inv(A)
        h = np.dot(A_inv,B)

        self.eigen_value, self.eigen_val_vector = np.linalg.eig(h)

        self.eigen_vector =  self.eigen_val_vector#[:3].reshape((20,3)) #unhash to use best 3 dominant feature vector

        # import ipdb;ipdb.set_trace()
        self.fisher_class_mean = np.array([np.dot(np.transpose(self.eigen_vector),i) for i in self.class_mean])
        self.fisher_class_cov_inv = np.array([np.linalg.inv(np.dot(np.dot(np.transpose(self.eigen_vector),i),self.eigen_vector)) for i in self.class_cov])

        return(True)

    def fisher_response(self,feature):
        g_resp = np.zeros((self.no_classes))

        f_feature = np.dot(np.transpose(self.eigen_vector),feature)
        
        for i in range(self.no_classes):
            vec = f_feature - self.fisher_class_mean[i]
            di = np.dot(np.dot(np.transpose(vec),self.fisher_class_cov_inv[i]),vec)
            g_resp[i]=float(di[0][0])

        resp = np.argmin(g_resp)

        return(resp)

    def test(self,data):
        cm = np.zeros((self.no_classes,self.no_classes)) 
        for idx,class_wise in enumerate(data):
            for feature in class_wise:

                #getting the actual and predicted results and appedning to results
                detected_class = self.fisher_response(feature)  
                actual_class = idx
                cm[actual_class,detected_class] += 1

        return(cm)



    def Fisher_classifier(self,train,test):
        if self.train == False:
            print('\n\nCreating the Fisher Space and Fisher Eigen Vector on training dataset...\n')
            trained = self.get_fisher(train)
            self.train = True
        else:
            print('Fisher Space already created...\n')

        confusion_matrix = self.test(test)
        print('Confusion Matrix is \n{}'.format(confusion_matrix))

        return(confusion_matrix)




# function to load the dataset
def load_cfar10_dataset(folder_path='../data/cifar_10', batch_id=1, len_test=1000):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # loading the dataset from the train_batch1 and test _batch from cifar10 dataset
    with open(folder_path + '/data_batch_' + str(batch_id), mode='rb') as train_file:
        train_batch = pickle.load(train_file, encoding='latin1')
    with open(folder_path + '/test_batch', mode='rb') as test_file:
        test_batch = pickle.load(test_file, encoding='latin1')
        
    #traing and test dataset
    train_set = (train_batch['data'].reshape((len(train_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1),np.array(train_batch['labels']))
    test_set = (test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1),np.array(test_batch['labels']))
    
    # finding length of class
    class_len = np.unique(test_set[1])

    #creating training dataset
    count_train = {i:0 for i in class_len}
    train = [[] for i in class_len]
    for idx,l in enumerate(train_set[1]):
        train[l].append(train_set[0][idx])
        count_train[l]+=1

    #creating testing dataset
    test = [[] for i in class_len]
    # create 100 datasets per class for testing
    count_test = {i:0 for i in class_len}
    len_test = int(len_test/len(count_test))
    for idx,l in enumerate(test_set[1]):
        if count_test[l]>=len_test:
            continue
        else:
            test[l].append(test_set[0][idx])
            count_test[l]+=1

    train_np = [np.array(batch) for batch in train]
    test_np = [np.array(batch) for batch in test]

    print('\nDataset per class, testing and training')
    for idx,(t1,t2) in enumerate(zip(count_train.values(),count_test.values())):
        print('Class-label :{}\t\t train-{}\t test-{}'.format(labels[idx],t1,t2))

    return (train_np, test_np, [count_train,count_test], labels)


# function for plotting the mean data of the fisher and actual.
def plot_mean(plt,feature_classifier):
    plt.subplot(212)
    ax1=plt.subplot(211)
    ax2=plt.subplot(212)

    #plotting the mean vec
    for i in range(feature_classifier.no_classes):
        ax1.plot(feature_classifier.class_mean[i], alpha=0.5, label = 'class-{}'.format(i))
    ax1.legend(loc=2)
    ax1.set_title('CIFAR mean data: Actual')

    #plotting the mean vector of fisher space
    for i in range(feature_classifier.no_classes):
        ax2.plot(feature_classifier.fisher_class_mean[i], alpha=0.5, label = 'class-{}'.format(i))
    ax2.legend(loc=2)
    ax2.set_title('CIFAR mean data: Fisher')

    return(plt)

if __name__ == '__main__':
    

    #loading the image and the mask; mask is sobel
    train,test,count,label = load_cfar10_dataset() 

    #craeting feature detector # (20,1) long feature vector
    feature_detector = harris_detector()
    print('Creating Harris features for training and testing images')
    train_features = feature_detector.create_features(train)
    test_features = feature_detector.create_features(test)

    #creating feature classifier using Fisher classifier object
    feature_classifier = fisher()
    cm = feature_classifier.Fisher_classifier(train=train_features,test=test_features)
    print('Accuracy:%d'%(np.sum(cm*np.eye(10))/np.sum(cm)*100))
    # creating the error rates
    error_rate = np.zeros(10) 
    for idx,i in enumerate(cm):
        error_rate[idx] = (np.sum(i)-i[idx])
    # printing the error rates
    for i,j in zip(error_rate,['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] ):
        print('Class-{}:{}%'.format(j,i))
    #plot of mean values of features : Fisher and Actual
    plt=plot_mean(plt,feature_classifier)
    plt.savefig('../data/mean_CIFAR_dataset.png')
    plt.show()

