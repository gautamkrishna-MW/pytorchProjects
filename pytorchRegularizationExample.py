

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as func
import matplotlib.pyplot as plt


targetDev = torch.device('cuda')
# a = torch.tensor([1,2,3,4,5,6,7,8])
# b = torch.tensor([1,9,3,4,5,5,7,8])
# out = (a==b)
# print(out.sum().item())

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784,512)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self, inp):
        inpFlat = inp.view(inp.shape[0], -1) # Flatten the input

        intOut0 = func.relu(self.fc0(inpFlat))
        intOut1 = func.relu(self.fc1(intOut0))
        intOut2 = func.relu(self.fc2(intOut1))
        intOut3 = func.relu(self.fc3(intOut2))
        output  = func.relu(self.fc4(intOut3))
        
        return output

class classifierWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784,512)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
        self.dpLayer = nn.Dropout(p=0.7)

    def forward(self, inp):
        inpFlat = inp.view(inp.shape[0], -1) # Flatten the input

        intOut0 = self.dpLayer(func.relu(self.fc0(inpFlat)))
        intOut1 = self.dpLayer(func.relu(self.fc1(intOut0)))
        intOut2 = self.dpLayer(func.relu(self.fc2(intOut1)))
        intOut3 = self.dpLayer(func.relu(self.fc3(intOut2)))
        output  = func.relu(self.fc4(intOut3))
        
        return output

print("Running NLLL")
# Download the dataset
trainData = datasets.MNIST(root='data', train=True,download=True)
testData = datasets.MNIST(root='data', train=False,download=True)

# Normalize the dataset 
testImg = trainData.data[0]
outMean, outStd = torch.mean(testImg.to(torch.float))/255, torch.std(testImg.to(torch.float))/255
transformObj_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(outMean,outStd)])
trainDataNorm = datasets.MNIST(root='data', train=True,transform=transformObj_norm)
testDataNorm = datasets.MNIST(root='data', train=False,transform=transformObj_norm)

# Check size
print(f"Dataset Size: {len(trainDataNorm)}")

# Creating DataLoader and splitting the train data 80-20
validationDataSize = 0.2 # 20% of training data
trainIdx = list(range(len(trainDataNorm)))
splitVal = int(np.round(len(trainDataNorm)*validationDataSize))
valDataIdx, trainDataIdx = trainIdx[:splitVal], trainIdx[splitVal:]

trainSampler = SubsetRandomSampler(trainDataIdx)
validSampler = SubsetRandomSampler(valDataIdx)

batchSize = 500
trainLoader = DataLoader(trainDataNorm,batch_size=batchSize,sampler=trainSampler)
validLoader = DataLoader(trainDataNorm,batch_size=batchSize,sampler=validSampler)
testLoader = DataLoader(testDataNorm,batch_size=batchSize)

# Check
imgs,labels = next(iter(trainLoader))
print("Train Size: ", imgs.shape)
print("Target Size: ", labels.shape)

imgs,labels = next(iter(validLoader))
print("Valid Size: ", imgs.shape)
print("Target Size: ", labels.shape)

imgs,labels = next(iter(testLoader))
print("Test Size: ", imgs.shape)
print("Target Size: ", labels.shape)

# plt.imshow(imgs[0][0])
# plt.pause(2)
# plt.show()
# plt.close()

# Creating model and running
# modelObj = classifier().cuda()
modelObj = classifierWithDropout().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(modelObj.parameters(),lr=0.1, weight_decay=0.01)

for epoch in range(0,31):
    train_loss = []
    valid_loss = []
    modelObj.train()
    for data, target in trainLoader:
        optimizer.zero_grad()

        output = modelObj(data.to(targetDev));
        loss = criterion(output,target.to(targetDev))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    with torch.no_grad():
        modelObj.eval()
        for vData, vTarget in validLoader:
            vOut = modelObj(vData.to(targetDev))
            vLoss = criterion(vOut,vTarget.to(targetDev))
            valid_loss.append(vLoss.item())
    print("Epoch: ", epoch, "Training Loss: ", np.mean(train_loss), "Validation Loss: ", np.mean(valid_loss))


# Testing the model
for data, target in testLoader:
    testOut = modelObj(data.to(targetDev))
    _,pred = torch.max(testOut,1)
    outVal = (target == pred.cpu()).sum()
    accuracy = outVal.item()/len(target)
    print("Accuracy: ", accuracy)

    
# # print("EndCode NLLLoss")