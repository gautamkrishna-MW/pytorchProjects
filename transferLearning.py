
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as pyModels
import kagglehub

def runMainCode():

    # For the Kaggle flower-dataset
    path = kagglehub.dataset_download("aritrase/flower-classification")
    print("Path to dataset files:", path)

    # Transforms for the input images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    # Dataset creation using "datasets.ImageFolder"
    img_datasets = {}
    img_datasets['train'] = datasets.ImageFolder(path + '/flowers_/train', train_transform)
    img_datasets['test'] = datasets.ImageFolder(path + '/flowers_/test', test_transform)
    print("class_names = ", img_datasets['train'].classes)
    numOutputClasses = len(img_datasets['train'].classes)

    # Creating data-loaders
    train_loader = torch.utils.data.DataLoader(img_datasets['train'],
                                                       batch_size=10,
                                                       shuffle=True,
                                                       num_workers=1)
    test_loader = torch.utils.data.DataLoader(img_datasets['test'],
                                                       batch_size=10,
                                                       shuffle=True,
                                                       num_workers=1)

    # Import and display model
    vgg16Model = pyModels.vgg16(pretrained=True)
    for param in vgg16Model.parameters():
        param.required_grad = False
    print(vgg16Model)

    # Get VGG16 model's 'classifier' section
    classifierInputs = vgg16Model.classifier[0].in_features

    # Create custom network for transfer learning
    newClassifier = nn.Sequential(nn.Linear(classifierInputs, numOutputClasses),
                                  nn.LogSoftmax(dim=1))

    # Replace new-classfier with old classifier in vgg-16
    vgg16Model.classifier = newClassifier
    print(vgg16Model)

    # Create loss function and optimizer
    criterion = nn.NLLLoss().cuda()
    optimizer = torch.optim.Adam(vgg16Model.classifier.parameters(), lr=0.001)
    vgg16Model.cuda()
    nEpochs = 10

    # Training
    for epochs in range(nEpochs):
        # monitor training loss
        train_loss = 0.0
        train_accuracy = 0
    
        ###################
        # train the model #
        ###################
        vgg16Model.train() # prep model for training
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = vgg16Model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            #calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epochs+1, 
                train_loss
                ))
        print(f"Train accuracy: {train_accuracy/len(train_loader):.3f}")

if __name__ == "__main__":
    runMainCode()