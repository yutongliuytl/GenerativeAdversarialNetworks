import torch
import torchvision

__all__ = [
    'Processing',
]


class Processing:
        
    def __init__(self):

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])


    #Hidden functions
    def __choose_dataset(self,dataset):

        if dataset == 'mnist':
            self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
        elif dataset == 'fashion_mnist':
            self.train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=self.transform, download=True)
        elif dataset == 'kmnist':
            self.train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, transform=self.transform, download=True)
        elif dataset == 'emnist':
            self.train_dataset = torchvision.datasets.EMNIST(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = torchvision.datasets.EMNIST(root='./data', train=False, transform=self.transform, download=True)
        elif dataset == 'qmnist':
            self.train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, transform=self.transform, download=True)
        

    #Callable functions
    def fit_transform(self,dataset,batch_size=256):
        
        self.__choose_dataset(dataset)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size ,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size ,shuffle=True)

        return train_loader,test_loader