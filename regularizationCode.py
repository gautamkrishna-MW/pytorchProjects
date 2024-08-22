
import numpy
import numpy as np
import torch
from matplotlib import pyplot
from torchvision.datasets import MNIST
from torchvision import transforms

# Creating Pytorch Network
from torch import nn, optim
import torch.nn.functional
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        hiddenLayerSizeArr = [512, 256, 128, 64]
        self.fc1 = nn.Linear(784,hiddenLayerSizeArr[0])
        self.fc2 = nn.Linear(hiddenLayerSizeArr[0],hiddenLayerSizeArr[1])
        self.fc3 = nn.Linear(hiddenLayerSizeArr[1], hiddenLayerSizeArr[2])
        self.fc4 = nn.Linear(hiddenLayerSizeArr[2], hiddenLayerSizeArr[3])
        self.fc5 = nn.Linear(hiddenLayerSizeArr[3], 10)
        self.dpLayer = nn.Dropout(p=0.3)

    def forward(self, inp):
        inp = inp.view(-1,784)
        layer1 = self.dpLayer(torch.nn.functional.relu(self.fc1(inp)))
        layer2 = self.dpLayer(torch.nn.functional.relu(self.fc2(layer1)))
        layer3 = self.dpLayer(torch.nn.functional.relu(self.fc3(layer2)))
        layer4 = self.dpLayer(torch.nn.functional.relu(self.fc4(layer3)))
        outputVal = self.fc5(layer4)
        return outputVal

# Check GPU
useGpu = torch.cuda.is_available()
if useGpu:
    targetDev = torch.device('cuda')
else:
    targetDev = torch.device('cpu')


# Download data and transform it to PyTorch Tensor
rawData = MNIST(root='data', download=True, transform=transforms.ToTensor())

# Normalize the rawData
sampleImg = rawData[numpy.random.randint(0,len(rawData),1)[0]]
avgVal = torch.mean(sampleImg[0])
stdVal = torch.std(sampleImg[0])
normalizationTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(avgVal, stdVal)])

# Creating a dataloaders for train and test datasets
from torch.utils.data import DataLoader
batchSize = 100
nrmData = MNIST(root='data', train=True, transform=normalizationTransforms)
nrmTestData = MNIST(root='data', train=False, transform=normalizationTransforms)

# Sampler for validation data
from torch.utils.data.sampler import SubsetRandomSampler
validSampleFraction = 0.2
valid_samples_size = int(numpy.round(validSampleFraction*len(nrmData)))
shuffled_idx = list(range(len(nrmData)))
numpy.random.shuffle(shuffled_idx)
nrmValidDataIdx, nrmTrainDataIdx = shuffled_idx[:valid_samples_size], shuffled_idx[valid_samples_size:]
nrmTrainDataSampler = SubsetRandomSampler(nrmTrainDataIdx)
nrmValidDataSampler = SubsetRandomSampler(nrmValidDataIdx)

# Creating dataloaders
trainLoader = DataLoader(nrmData, batch_size=batchSize, num_workers=0, sampler=nrmTrainDataSampler, pin_memory=True, pin_memory_device='cuda')
validLoader = DataLoader(nrmData, batch_size=batchSize, num_workers=0, sampler=nrmValidDataSampler, pin_memory=True, pin_memory_device='cuda')
testLoader = DataLoader(nrmTestData, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device='cuda')

# Create Network
modelObj = Model()
modelObj = modelObj.to(targetDev)
optimObj = torch.optim.SGD(modelObj.parameters(), lr=0.01, weight_decay=0.01)

if useGpu:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epIdx in range(0,num_epochs):
    train_loss = []
    valid_loss = []

    modelObj.train()
    for inp,target in trainLoader:
        optimObj.zero_grad()

        outVal = modelObj(inp.to(targetDev))
        loss = criterion(outVal, target.to(targetDev))
        loss.backward()
        optimObj.step()
        train_loss.append(loss.item())

    with torch.no_grad():
        modelObj.eval()
        for v_inp, v_target in validLoader:
            v_outVal = modelObj(v_inp.to(targetDev))
            v_loss = criterion(v_outVal, v_target.to(targetDev))
        valid_loss.append(v_loss.item())
    print(f"Epoch {epIdx} Mean Validation Loss: {np.mean(valid_loss)}")
    print(f"Epoch: {epIdx}, Mean Training Loss: {np.mean(train_loss)}")


# Prediction
scoreVal = 0
for inp, target in testLoader:
    predVal = torch.argmax(torch.exp(modelObj(inp.to(targetDev))), dim=1)
    [print(f"Actual:{target[i]}, Predicted:{predVal[i]}") for i in range(batchSize)]