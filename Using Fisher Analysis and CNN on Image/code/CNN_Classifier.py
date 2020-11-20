'''

Assignment 4 for Computer Vision- CAP 5415.
Question 2: CNN based classifier
Abraham Jose
11-14-2019
abraham@knights.ucf.edu 

'''
# from __future__ import print_function
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import time
import torchvision.utils as vutils
from torchsummary import summary as model_summary

from torch.utils.tensorboard import SummaryWriter

# class for network as described in the question
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.bn_input = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
		self.fc1 = nn.Linear(64*1*1,64)
		self.fc2 = nn.Linear(64,10)

	# forward path of the model
	def forward(self, x):
		x = self.bn_input(x)
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)

		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(x, 2, 2)

		x = x.view(-1,64*1*1)
		
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

	# def summary(self):
	# 	from torchsummary import summary
	# 	summary(self,input_size=(3,32,32))



# class for training the model
class train_model:
	# initilizing the model params and details
	def __init__(self,log_interval, model, device, optimizer, epochs, log=None, train = True):
		self.log_interval = log_interval
		self.model = model
		self.device = device
		# print(self.device)
		if train == True:
			self.optimizer = optimizer
		self.epochs = epochs
		self.current_epoch = 0
		if log != None:
			self.log = log

	# training code for the model on train_loader dataset generator
	def train(self,train_loader):
		

		# setting model for training
		self.model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data_dev)
			# using cross-entropy as loss function for training
			loss =  F.cross_entropy(output, target.view_as(target))
			loss.backward()
			self.optimizer.step()

			# _, argmax = torch.max(output, 1)
			# accuracy = (target == argmax.squeeze()).float().mean()

			# for logging at the logging interval during the epoch
			if batch_idx % self.log_interval == 0:
				
				self.current_log_interval+=1

				print('Train Epoch: {} --- {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					self.current_log_interval, self.current_epoch, batch_idx * len(data_dev), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))
				
				self.log.add_scalar('training_loss',loss.item(),self.current_log_interval)

		return(True)

	def val(self, val_loader):

		# setting model for validation
		self.model.eval()
		val_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in val_loader:
				data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
				output = self.model(data_dev)
				val_loss += F.cross_entropy(output, target.view_as(target), reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		# finding validation loss and accuracy and logging them
		val_loss /= self.val_len
		accuracy = 100. * correct / self.val_len

		self.log.add_scalar('validataion_loss',val_loss,self.current_epoch)
		self.log.add_scalar('validataion_accuracy',accuracy,self.current_epoch)

		print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			val_loss, correct, self.val_len, accuracy))
		return(True,accuracy)

	def test(self, test_loader):
		self.test_len = len(test_loader.dataset)
		# setting model for testing
		self.model.eval()
		test_loss = 0
		correct = 0
		# creating model for testing and confusion matrix
		cm = np.zeros((10,10))
		with torch.no_grad():
			for data, target in test_loader:
				data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
				output = self.model(data_dev)
				test_loss += F.cross_entropy(output, target.view_as(target), reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()
				for actual,predicted in zip(target.view_as(pred),pred):
					cm[actual,predicted]+=1

		test_loss /= self.test_len
		accuracy = 100. * correct / self.test_len
		return(True,cm,accuracy)


	def iterate(self,train_loader, val_loader, test_loader):
		self.train_len = len(train_loader.dataset)
		self.val_len = len(val_loader.dataset)
		self.batch_size=train_loader.batch_size
		self.current_log_interval=0
		self.val_accuracy_best = 0
		self.test_cm_best = np.zeros((10,10))
		self.len = train_loader

		print("training dataset size = {}\ntesting dataset size = {}\nbatch size = {}".format(self.train_len,self.val_len,self.batch_size))
		# for iteration for training, testing and validation batch
		test_result = []
		for epoch in range(1, self.epochs + 1):
			train_flag = self.train(train_loader = train_loader)
			if train_flag != True:
				return(False,epoch)
			val_flag,accuracy = self.val(val_loader = val_loader)
			if val_flag != True:
				return(False,epoch)
		
			if self.val_accuracy_best < accuracy:
				self.save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'best_accuracy': accuracy},
					filename='../data/checkpoint/best_test-epoch:{}-accuracy:{}.pth.tar'.format(epoch,accuracy))
				torch.save(model.state_dict(), "../data/model/bestx_model.pt")
				val_flag,cm,accuracy = self.test(test_loader = test_loader)
				self.test_cm_best = cm
				print('Confusion Matrix:\n',cm)
				self.val_accuracy_best = accuracy
			# print(self.test_accuracy_best)
			
			test_result.append(accuracy)
			self.current_epoch+=1

		return(True,test_result)

	def save_checkpoint(self,state, filename='/output/checkpoint.pth.tar'):
		"""Save checkpoint if a new best is achieved"""
		print ("=> Saving a new best")
		torch.save(state, filename)  # save checkpoint

#dataset class for cifar10
class CifarDataset(Dataset):
	def __init__(self, data, transform = None):
		self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		self.X = data[0]
		self.Y = data[1]
		self.size = len(data[1])
		u,c = np.unique(data[1],return_counts=True)
		self.dist = {i:j for i,j in zip(u,c)}
		self.transform = transform
	
	# create a random sample of dataset for testing.
	def samples(self,sample_size=10):
		sample = [self.__getitem__(np.random.randint(0,self.size)) for i in range(sample_size)]
		return(sample)

	# for visualization of the training and testing dataset after augmentation
	def sample_image(self,sample_size=(2,2)):
		import matplotlib.pyplot as plt
		c=0
		for i in range(sample_size[0]):
			for j in range(sample_size[1]):
				c+=1
				axis = plt.subplot(sample_size[0],sample_size[1],c)
				item = self.__getitem__(np.random.randint(0,self.size))
				if type(item[0]) == torch.Tensor:
					axis.imshow(transforms.functional.to_pil_image(item[0]))
				else:
					axis.imshow(item[0]) 
				axis.set_title("{}: {}".format(str(item[1]),self.labels[item[1]])) 
				plt.yticks([])
				plt.xticks([])
		plt.show()

	def __len__(self):
		self.size = len(self.Y)
		return (self.size)
	
	# method for getting the dataset
	def __getitem__(self, idx):
		item = self.X[idx]
		label = self.Y[idx]
		if self.transform != None:
			item = self.transform(item)
		return (item, label);

# function to load the dataset
def load_cfar10_dataset(transform=None,folder_path='../data/cifar_10', batch_id=1, len_test=1000):

	# loading the dataset from the train_batch1 and test _batch from cifar10 dataset
	with open(folder_path + '/data_batch_' + str(batch_id), mode='rb') as train_file:
		train_batch = pickle.load(train_file, encoding='latin1')
	with open(folder_path + '/test_batch', mode='rb') as test_file:
		test_batch = pickle.load(test_file, encoding='latin1')
		
	#traing and test dataset
	train = (train_batch['data'].reshape((len(train_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1),np.array(train_batch['labels']))
	test_set = (test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1),np.array(test_batch['labels']))
	
	# create 100 datasets per class for testing
	count = {i:0 for i in np.unique(test_set[1])}
	len_test = int(len_test/len(count))
	test = [[],[]]
	for idx,l in enumerate(test_set[1]):
		if count[l]>=len_test:
			continue
		else:
			test[0].append(test_set[0][idx])
			test[1].append(l)
			count[l]+=1
	#if 2 transforms are available, use train and test augmentation seperately
	if len(transform) == 2:
		return (CifarDataset(train,transform = transform[0]), CifarDataset(test,transform = transform[1]))
	else:
		return (CifarDataset(train,transform = transform), CifarDataset(test,transform = transform))



if __name__ == '__main__':
	# training arguments
	parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10000, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--log-interval', type=int, default=20, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	parser.add_argument('--test_model', default=False,
						help='For testing the current Model')
	args = parser.parse_args()

	#random seeding the training
	torch.manual_seed(1)

	# opting GPU if GPU is available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	# creating the augmentation for the dataset
	transform_train = transforms.Compose([transforms.ToPILImage(mode='RGB'),
									transforms.RandomApply([
									transforms.ColorJitter(brightness=.2, contrast=.5, saturation=.5, hue=0.1),
									transforms.RandomAffine(degrees=20, translate=(0.2,0.2), scale=None, shear=(.1,.1), resample=False, fillcolor=0),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip(),
									], p =0.25),
									transforms.ToTensor(),
									])
	transform_test = transforms.Compose([transforms.ToPILImage(mode='RGB'),
									transforms.ToTensor(),
									])

	#loading the cifar10 datset
	train,test = load_cfar10_dataset([transform_train,transform_test],folder_path = '../data/cifar_10/')

	# creating training,testing and validation datasets
	train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=args.batch_size,
											   shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=args.batch_size,
											   shuffle=True);
	val_loader = torch.utils.data.DataLoader(dataset=test,batch_size=args.batch_size,
											   shuffle=False);

	# creating the model to CPU
	device ='cpu'
	model = Net().to(device)
	print('\nModel Summary:\n',model_summary(model,input_size=(3,32,32),device =device))

	if args.test_model == False:
		# training the model

		#logging of training
		log = SummaryWriter('../data/model/runs/')
		images,label = next(iter(train_loader))
		grid = vutils.make_grid(images)
		log.add_image('images',grid)
		log.add_graph(model,images)

		# initilizing the optimizer
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

		# creating the training object and iterating each epoch
		train = train_model(args.log_interval, model, device, optimizer, args.epochs, log =log)
		print(train.device)
		train.iterate(train_loader = train_loader,val_loader = val_loader, test_loader =test_loader)

		print('\n\n\n\n\nFinal confusion matrix :\n\n',train.test_cm_best)	
		# saving the model
		if args.save_model:
			torch.save(model.state_dict(), "../data/model/final_model.pt")
		log.close()
		print('Completed training')
	else:
		# testing the model 
		model.load_state_dict(torch.load("../data/model/best_model.pt"))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
		# careating the mdoel testing instance
		test = train_model(args.log_interval, model, device, optimizer, args.epochs, train= False)
		# getting the confusion matrix
		val_flag,cm,accuracy = test.test(test_loader = test_loader)
		print("\nConfusion Matrix:\n",cm)
		print('Accuracy:%d'%(np.sum(cm*np.eye(10))/np.sum(cm)*100))

		# creating the error rates
		error_rate = np.zeros(10) 
		for idx,i in enumerate(cm):
			error_rate[idx] = (np.sum(i)-i[idx])
		# printing the error rates
		for i,j in zip(error_rate,['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] ):
			print('Class-{}:{}%'.format(j,i))

		print('Completed testing the best model')





