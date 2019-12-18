import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision.transforms as transforms
from torch.autograd import Variable, Function

import os
import numpy as np
import time
import sys


# Comment out for cpu operation
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]='1'
#################################
import json
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import copy

from cnn import Net1, Net2


def maketensor(image_array, labels_array):
    
    image_array = image_array.reshape([image_array.shape[0] ,3, 80, 80])

    image_array = image_array/255.0

    image_tensor = torch.from_numpy(image_array)

    # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # tensor_normalized = torch.zeros_like(image_tensor)

    # for i in range(image_tensor.shape[0]):
    #     tensor_normalized[i] = normalize(image_tensor[i])


    labels_tensor = torch.from_numpy(labels_array)
    labels_tensor = labels_tensor.type(torch.FloatTensor)

    return image_tensor, labels_tensor




def calculate_metrics(prediction, ground_truth, tensor_size):

    corrects = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    #Accuracy calculation
    corrects += ((prediction==ground_truth).sum(0)).float()

    for k in range(tensor_size):

        if (prediction[k]==1 and ground_truth[k]==1):
            true_positive += 1
        elif (prediction[k]==1 and ground_truth[k]==0):
            false_positive += 1
        elif (prediction[k]==0 and ground_truth[k]==0):
            true_negative += 1
        elif (prediction[k]==0 and ground_truth[k]==1):
            false_negative += 1

    return corrects, true_positive, false_positive, true_negative, false_negative

batchsize=16

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

model = Net2()
model.to(device)
state = torch.load('../results/previous_results/trainedNetBest-epoch-463-acc-0.9919.pt')	
# print(state)
model.load_state_dict(state)
model.eval()

input_array = torch.load('../dataset/sample_array.pth')
labels_array = torch.load('../dataset/sample_labels.pth')

input_tensor, label_tensor = maketensor(input_array, labels_array)

testsize = int(input_tensor.shape[0])

print('Number of sample images :',testsize)

batchID = 1

corrects = 0

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

detected=[]
actual=[]

with torch.no_grad():      

    for i in range(0,testsize,batchsize):

        # if batchID%25==0:
        #     print("batchID : "+str(batchID)+'/'+str(testsize/batchsize))

        if i+batchsize<=testsize:
            inputs = input_tensor[i:i+batchsize]
            labels = label_tensor[i:i+batchsize]

        else:
            inputs = input_tensor[i:]
            labels = label_tensor[i:]

        inputs, labels = Variable(inputs.float().to(device)), Variable(labels.to(device))                       

        # Feed-forward
        output = model(inputs)

        _, predicted = torch.max(output.data, 1)
        _, targetindex = torch.max(labels.data, 1)

        # if predicted==1:
        # 	d='Ship'
        # 	detected.append('Ship')
        # else:
        # 	d='No_Ship'
        # 	detected.append('No_Ship')

        # if targetindex==1:
        # 	a='Ship'
        # 	actual.append('Ship')
        # else:
        # 	a='No_Ship'
        # 	actual.append('No_ship')

        # print('Image {}: Actual:{}; Detected:{}'.format(batchID, a, d))

        if i+batchsize<=testsize:

            size_tensor = batchsize

        else:

            size_tensor = testsize%batchsize

        C, TP, FP, TN, FN = calculate_metrics(predicted, targetindex, size_tensor)

        corrects += C
        true_positive += TP
        false_positive += FP
        true_negative += TN
        false_negative += FN
        #print(corrects)

        batchID += 1

sensitivity = true_positive/float(true_positive+false_negative)
specificity = true_negative/float(true_negative+false_positive)
accuracy = corrects/float(testsize)

print('Test Accuracy : {:.6f} ; Test Sensitivity : {:.6f} ; Test Specificity : {:.6f}'.format(accuracy, sensitivity, specificity))

torch.save(actual, '../dataset/GT')
print(len(actual))
torch.save(detected, '../dataset/Predicted')
print(len(detected))