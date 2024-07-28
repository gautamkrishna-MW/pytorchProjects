'''
Datasets and Dataloaders in Pytorch
'''
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch

# Datasets
x = torch.rand(3,4)
print(x)
dataObj = DataLoader(x)
for items in dataObj:
    for newItems in items:
        print(newItems)


z = torch.flatten(x)
print(z)
newDataloader = DataLoader(z,batch_size=5, shuffle=True)
for items in newDataloader:
    print(items)

from sklearn.datasets import make_classification
X,y = make_classification(n_samples=100)
print([X.shape, y.shape])
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

dataLoader = DataLoader(dataset, batch_size=5)
for i in dataLoader:
    print(i)
