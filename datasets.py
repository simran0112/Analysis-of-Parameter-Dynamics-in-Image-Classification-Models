#dataset loaders 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize, Pad

import os 
def return_dataset():      
  transform=Compose([
    Pad(2),
    ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1))
    ])

  fashionmnist_train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True, 
    transform=transform   
  )


  fashionmnist_test = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=transform
  )

  fashionmnist_labels = fashionmnist_test.classes
  fashionmnist_train.name = "fmnist_trn"
  fashionmnist_test.name = 'fmnist_tst'
  if os.path.exists("MNIST.tar.gz"):
    os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz')
    os.system('tar -zxvf MNIST.tar.gz')
    os.system('rm MNIST.tar.gz')
  mytransform=Compose([
    Pad(2),
    ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1))       

    ])

  mnist_train = MNIST(root = './', train=True, download=False, transform=mytransform)
  mnist_test = MNIST(root = './', train=False, download=False, transform=mytransform)

  # name mnist
  mnist_train.name = 'mnist_trn'
  mnist_test.name = 'mnist_tst'



  # cifar10 32*32*3
  cifar10_train = datasets.CIFAR10(
      root="data",
      train=True,
      download=True,
      transform=ToTensor()
  )

  cifar10_test = datasets.CIFAR10(
      root="data",
      train=False,
      download=True,
      transform=Compose([
        ToTensor(),
      ]) # input from Satwik?
  )

  cifar10_train.targets = torch.tensor(cifar10_train.targets)
  cifar10_test.targets = torch.tensor(cifar10_test.targets)

  cifar10_train.name = 'cifar_trn'
  cifar10_test.name = 'cifar_tst'

  return [[mnist_train, mnist_test], [fashionmnist_train, fashionmnist_test], [cifar10_train, cifar10_test]]

