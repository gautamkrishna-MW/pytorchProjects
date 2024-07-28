'''
MNIST handwriiten digit recognition
using GPU and advanced model
'''
import numpy as np
import numpy.random
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from matplotlib import pyplot

# Creating Pytorch Network
from torch import nn, optim
import torch.nn.functional
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        hiddenLayerSizeArr = [256, 128, 64]
        self.fc1 = nn.Linear(784,hiddenLayerSizeArr[0])
        self.fc2 = nn.Linear(hiddenLayerSizeArr[0],hiddenLayerSizeArr[1])
        self.fc3 = nn.Linear(hiddenLayerSizeArr[1], hiddenLayerSizeArr[2])
        self.fc4 = nn.Linear(hiddenLayerSizeArr[2], 10)

    def forward(self, inp):
        inp = inp.view(-1,784)
        layer1 = torch.nn.functional.relu(self.fc1(inp))
        layer2 = torch.nn.functional.relu(self.fc2(layer1))
        layer3 = torch.nn.functional.relu(self.fc3(layer2))
        outputVal = torch.nn.functional.log_softmax(self.fc4(layer3), dim=1)
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
nrmData = MNIST(root='data', train=True, transform=normalizationTransforms)

# Display sample Image
pyplot.imshow(sampleImg[0].reshape([28,28]))
pyplot.title(f"Number [avg,var]: {sampleImg[1]} [{avgVal:.2f}, {stdVal:.2f}]")
pyplot.show(block=False)
pyplot.pause(2)
pyplot.close()

# Creating a dataloader
from torch.utils.data import DataLoader
trainLoader = DataLoader(nrmData, batch_size=100, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device='cuda')

[img, label] = next(iter(trainLoader))
img2 = img.view(-1,784)
print(img.shape)
print(img2.shape)

# Create Network
modelObj = Model()
modelObj = modelObj.to(targetDev)
optimObj = torch.optim.Adam(modelObj.parameters(), lr=0.01)

if useGpu:
    criterion = nn.NLLLoss().cuda()
else:
    criterion = nn.NLLLoss()

num_epochs = 10
for epIdx in range(0,num_epochs):
    train_loss = []

    for inp,target in trainLoader:
        optimObj.zero_grad()

        outVal = modelObj(inp.to(targetDev))
        loss = criterion(outVal, target.to(targetDev))
        loss.backward()
        optimObj.step()
        train_loss.append(loss.item())

    print(f"Epoch: {epIdx}, Loss: {np.mean(train_loss)}")


# Prediction
nrmTestData = MNIST(root='data', train=False, transform=normalizationTransforms)
testLoader = DataLoader(nrmTestData, batch_size=100, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device='cuda')

for inp, target in testLoader:
    predVal = torch.argmax(torch.exp(modelObj(inp.to(targetDev))), dim=1)
    [print(f"Actual:{target[i]}, Predicted:{predVal[i]}") for i in range(100)]

