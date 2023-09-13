#@author: Manish Bhattarai, Mehmet Cagri

import torch
from torch.utils import data
import numpy as np
import torchvision
import torchvision.transforms as transforms

class DatasetLoader():
    """
    A utility class to load different types of datasets for machine learning experiments.

    ...

    Attributes
    ----------
    name : str
        The name of the dataset to be loaded
    params : dict
        A dictionary containing hyperparameters for the dataloader

    Methods
    -------
    preprocess_data(mean, std):
        Applies standard preprocessing transformations to the dataset.
    cifar10_dataload():
        Loads the CIFAR-10 dataset.
    cifar100_dataload():
        Loads the CIFAR-100 dataset.
    MNIST_dataload():
        Loads the MNIST dataset.
    FashionMNIST_dataload():
        Loads the FashionMNIST dataset.
    Imagenet_dataload():
        Loads the ImageNet dataset.
    fit():
        Calls the appropriate dataload method based on the dataset name.
    kfold_split(dataset, transform_train=None, transform_valid=None):
        Splits the dataset into training and validation sets for k-fold cross-validation.
    """

    def __init__(self,name='cifar10',params={'batch_size': 64, 'num_workers': 16, 'shuffle': True,'normalize':True, 'nfolds':1}):
        """
        Constructor to initialize the DatasetLoader.

        Parameters
        ----------
        name : str
            The name of the dataset to be loaded.
        params : dict
            A dictionary of hyperparameters for dataset loading.
        """
        self.params =params
        self.name = name

    def preprocess_data(self,mean,std):
        """
        Applies standard preprocessing transformations to the dataset.

        Parameters
        ----------
        mean : tuple or float
            The mean for each channel for normalization.
        std : tuple or float
            The standard deviation for each channel for normalization.

        Returns
        -------
        transform_train : torchvision.transforms.Compose
            The transformations to apply to the training set.
        transform_test : torchvision.transforms.Compose
            The transformations to apply to the test set.
        """
        print('==> Preparing data..')
        if self.params['normalize']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std ),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize(mean,std ),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
                ])

        return transform_train,transform_test

    def cifar10_dataload(self):
        """
        Loads the CIFAR-10 dataset using the torchvision library and applies preprocessing transformations.

        Returns
        -------
        trainloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        testloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        classes : tuple
            Tuple containing class names.
        mean : tuple
            Tuple containing channel-wise mean for normalization.
        std : tuple
            Tuple containing channel-wise standard deviation for normalization.
        """
        # Data
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        transform_train, transform_test = self.preprocess_data(mean,std)
        transform_train_cur = transform_train if self.params['nfolds'] == 1 else None

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train_cur)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.params['batch_size'], shuffle=False,
            num_workers=self.params['num_workers'])

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        if self.params['nfolds'] == 1:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params['batch_size'], shuffle=self.params['shuffle'],
                num_workers=self.params['num_workers'])

            return trainloader, testloader, classes, mean, std 
        else:
            splits = self.kfold_split(trainset, transform_train=transform_train,
                             transform_valid=transform_test)
            return splits, testloader, classes, mean, std 

    def cifar100_dataload(self):
        """
        Loads the CIFAR-100 dataset using the torchvision library and applies preprocessing transformations.

        Returns
        -------
        trainloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        testloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        classes : tuple
            Tuple containing class names.
        mean : tuple
            Tuple containing channel-wise mean for normalization.
        std : tuple
            Tuple containing channel-wise standard deviation for normalization.
        """
        # Data
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        transform_train, transform_test = self.preprocess_data(mean, std)
        transform_train_cur = transform_train if self.params['nfolds'] == 1 else None

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train_cur)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.params['batch_size'], shuffle=False,
            num_workers=self.params['num_workers'])

        # TODO: find the actual class lables
        classes = list(range(100))

        if self.params['nfolds'] == 1:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params['batch_size'], shuffle=self.params['shuffle'],
                num_workers=self.params['num_workers'])

            return trainloader, testloader, classes, mean, std 
        else:
            splits = self.kfold_split(trainset, transform_train=transform_train,
                             transform_valid=transform_test)
            return splits, testloader, classes, mean, std 

    def MNIST_dataload(self):
        """
        Loads the MNIST dataset using the torchvision library and applies preprocessing transformations.

        Returns
        -------
        trainloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        testloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        classes : tuple
            Tuple containing class names.
        mean : float
            Mean for grayscale normalization.
        std : float
            Standard deviation for grayscale normalization.
        """
        # Data
        mean = 0.1307
        std = 0.3081

        transform_train, transform_test = self.preprocess_data(mean, std)
        transform_train_cur = transform_train if self.params['nfolds'] == 1 else None

        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train_cur)

        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.params['batch_size'], shuffle=False,
            num_workers=self.params['num_workers'])

        classes = (0,1,2,3,4,5,6,7,8,9)
        if self.params['nfolds'] == 1:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params['batch_size'], shuffle=self.params['shuffle'],
                num_workers=self.params['num_workers'])

            return trainloader, testloader, classes, mean, std 
        else:
            splits = self.kfold_split(trainset, transform_train=transform_train,
                             transform_valid=transform_test)
            return splits, testloader, classes, mean, std 

    def FashionMNIST_dataload(self):
        """
        Loads the FashionMNIST dataset using the torchvision library and applies preprocessing transformations.

        Returns
        -------
        trainloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        testloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        classes : tuple
            Tuple containing class names.
        mean : float
            Mean for grayscale normalization.
        std : float
            Standard deviation for grayscale normalization.
        """
        # Data
        mean = 0.2860
        std = 0.3530

        transform_train, transform_test = self.preprocess_data(mean, std)
        transform_train_cur = transform_train if self.params['nfolds'] == 1 else None

        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform_train_cur)

        testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.params['batch_size'], shuffle=False,
            num_workers=self.params['num_workers'])

        classes = (0,1,2,3,4,5,6,7,8,9)
        if self.params['nfolds'] == 1:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params['batch_size'], shuffle=self.params['shuffle'],
                num_workers=self.params['num_workers'])

            return trainloader, testloader, classes, mean, std 
        else:
            splits = self.kfold_split(trainset, transform_train=transform_train,
                             transform_valid=transform_test)
            return splits, testloader, classes, mean, std 

    def Imagenet_dataload(self):
        """
        Loads the ImageNet dataset using the torchvision library and applies preprocessing transformations.

        Returns
        -------
        trainloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        testloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        classes : tuple
            Tuple containing class names.
        mean : tuple
            Tuple containing channel-wise mean for normalization.
        std : tuple
            Tuple containing channel-wise standard deviation for normalization.
        """
        # Data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_train, transform_test = self.preprocess_data(mean, std)
        transform_train_cur = transform_train if self.params['nfolds'] == 1 else None

        trainset = torchvision.datasets.ImageNet(
            root='./data', train=True, download=True, transform=transform_train_cur)

        testset = torchvision.datasets.Imagenet(
            root='./data', train=False, download=True, transform=transform_test)
        # TODO: find the actual class lables
        classes = list(range(1000))
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.params['batch_size'], shuffle=False,
            num_workers=self.params['num_workers'])
        if self.params['nfolds'] == 1:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params['batch_size'], shuffle=self.params['shuffle'],
                num_workers=self.params['num_workers'])

            return trainloader, testloader, classes, mean, std
        else:
            splits = self.kfold_split(trainset, transform_train=transform_train,
                             transform_valid=transform_test)
            return splits, testloader, classes, mean, std

    def fit(self):
        """
        Calls the appropriate dataload method based on the dataset name specified during initialization.

        Returns
        -------
        A DataLoader object for the chosen dataset.
        """
        self.name = self.name.lower()
        if self.name == 'cifar10':
            return self.cifar10_dataload()
        elif self.name == 'cifar100':
            return self.cifar100_dataload()
        elif self.name == 'mnist':
            return self.MNIST_dataload()
        elif self.name == 'fmnist':
            return self.FashionMNIST_dataload()
        elif self.name == 'imagenet':
            return self.Imagenet_dataload()
        else:
            raise ValueError("Dataset not found")

    def kfold_split(self, dataset, transform_train=None, transform_valid=None):
        """
         Splits the dataset into training and validation sets for k-fold cross-validation.

         Parameters
         ----------
         dataset : torch.utils.data.Dataset
             The dataset to be split.
         transform_train : torchvision.transforms.Compose, optional
             Transformations to be applied on training dataset.
         transform_valid : torchvision.transforms.Compose, optional
             Transformations to be applied on validation dataset.

         Returns
         -------
         all_loaders : list of tuple
             List of tuples where each tuple contains a DataLoader for the training set and a DataLoader for the validation set for each fold.

        """
        from sklearn.model_selection import KFold
        from subset import Subset
        # do not shuffle kfold for reproducibility
        folds = KFold(n_splits=self.params['nfolds'], shuffle=False)
        all_loaders = []
        for i_fold, (train_idx, valid_idx) in enumerate(folds.split(dataset)):
            dataset_train = Subset(dataset, train_idx, transform_train)
            dataset_valid = Subset(dataset, valid_idx, transform_valid)
            trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.params['batch_size'],
                                                      shuffle=self.params['shuffle'],
                                                      num_workers=self.params['num_workers'],
                                                      pin_memory=True)
            validloader = torch.utils.data.DataLoader(dataset_valid, batch_size=self.params['batch_size'],
                                                      shuffle=False,
                                                      num_workers=self.params['num_workers'],
                                                      pin_memory=True)
            all_loaders.append((trainloader, validloader))
        return all_loaders


