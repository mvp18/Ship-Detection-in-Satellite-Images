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
##############################
import json
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import copy

from cnn import Net1, Net2

#Network Settings

def tensorshuffle(it, il, N):
        
    for loop in range(int(N/2)):

        irand1 = np.random.randint(0, N)
        irand2 = np.random.randint(0, N)

        ivar = it[irand1].clone()
        it[irand1] = it[irand2]
        it[irand2] = ivar
        
        ivar = il[irand1].clone()
        il[irand1] = il[irand2]
        il[irand2] = ivar

    return it, il

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

save_path = '../results'

if not os.path.exists(save_path):
    os.makedirs(save_path)

batchsize = 16
num_epochs = 1000
learning_rate = 0.001
# ---------------

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

model = Net2()
model.to(device)

# Dataset Preparation

# f = open('../dataset/shipsnet.json')
# dataset = json.load(f)
# f.close()

# labels = np.array(dataset['labels']).astype('uint8')
# image = np.array(dataset['data']).astype('uint8')

print('\nLoading data')

image = torch.load('../dataset/ship_array.pth')
labels = torch.load('../dataset/labels_array.pth')

image_train, image_test, labels_train, labels_test = train_test_split(image, labels, test_size = 0.25)

print('\nMaking tensors')

train_tensor, train_label_tensor = maketensor(image_train, labels_train)
test_tensor, test_label_tensor = maketensor(image_test, labels_test)

# Loss and Optimizer

criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, nesterov = True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

#Training

def trainepoch(model, input_tensor, label_tensor, criterion, optimizer, batchsize):

    print('\nTraining')

    model.train()

    runningLoss = 0 

    trainsize = int(input_tensor.shape[0])

    batch_counter = 0

    batchID = 1        
    
    for i in range(0,trainsize,batchsize):

        if batchID%50==0:
            print("batchID : "+str(batchID)+'/'+str(trainsize/batchsize))

        if i+batchsize<=trainsize:
            inputs = input_tensor[i:i+batchsize]
            labels = label_tensor[i:i+batchsize]

        else:
            inputs = input_tensor[i:]
            labels = label_tensor[i:]

        inputs, labels = Variable(inputs.float().to(device)), Variable(labels.to(device))                       

        # Feed-forward
        output = model(inputs)
        # Compute loss/error
        #loss = criterion(output, torch.max(labels, 1)[1]) - for CrossEntropy Loss
        loss = criterion(output, labels)
        # Accumulate loss per batch
        runningLoss += loss.data.cpu()
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()

        batchID += 1
        batch_counter += 1

    return(model, runningLoss/float(batch_counter))


def testNet(model, input_tensor, label_tensor, batchsize):

    print('\nTesting')

    model.eval()

    testsize = int(input_tensor.shape[0])

    batchID = 1

    corrects = 0.0

    true_positive = 0.0
    false_positive = 0.0
    true_negative = 0.0
    false_negative = 0.0

    with torch.no_grad():      
    
        for i in range(0,testsize,batchsize):

            if batchID%50==0:
                print("batchID : "+str(batchID)+'/'+str(testsize/batchsize))

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

    return(corrects/float(testsize), sensitivity, specificity)


trainLoss = []
testaccuracy = []
test_sensitivity = []
test_specificity = []

best_accuracy = 0.0

for epochNum in range(num_epochs): 

    if (epochNum+1)%300==0:
        optimizer.param_groups[0]['lr']/=10
        print('New learning rate :', optimizer.param_groups[0]['lr'])
    
    epochStart = time.time()
    shuffled_image, shuffled_label = tensorshuffle(train_tensor, train_label_tensor, int(train_tensor.shape[0]))
    net, avgTrainLoss = trainepoch(model, shuffled_image, shuffled_label, criterion, optimizer, batchsize)
    model_accuracy, model_sensitivity, model_specificity = testNet(net, test_tensor, test_label_tensor, batchsize)
    trainLoss.append(avgTrainLoss)
    testaccuracy.append(model_accuracy)
    test_specificity.append(model_specificity)
    test_sensitivity.append(model_sensitivity)

    # Saving model when accuracy is maximum
    if model_accuracy>best_accuracy:
        trained_model = copy.deepcopy(net.state_dict())
        print('\nNew best model')
        print('Epoch {:.0f}: saving model'.format(epochNum+1))
        best_accuracy = model_accuracy
        bestEp = epochNum+1

    # Saving train and validation loss
    with open(os.path.join(save_path,'trainLoss.pkl'),'wb') as f:
        pickle.dump(trainLoss,f)
    with open(os.path.join(save_path,'testaccuracy.pkl'),'wb') as f:
        pickle.dump(testaccuracy,f)

    with open(os.path.join(save_path,'test_sensitivity.pkl'),'wb') as f:
        pickle.dump(test_sensitivity,f)

    with open(os.path.join(save_path,'test_specificity.pkl'),'wb') as f:
        pickle.dump(test_specificity,f)


    epochEnd = time.time()-epochStart   
    print('\nIteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f} ; Accuracy : {:.6f} ; Sensitivity : {:.6f} ; Specificity : {:.6f} ; Time consumed: {:.0f}m {:.0f}s '\
          .format(epochNum + 1,num_epochs,avgTrainLoss, model_accuracy, model_sensitivity, model_specificity, epochEnd//60,epochEnd%60))


print('\nBest performance at epoch '+str(bestEp))
best_model = trained_model
torch.save(best_model,os.path.join(save_path,'trainedNetBest'+ '-epoch-' + str(bestEp) + '-acc-' + str(best_accuracy) + '.pt'))
