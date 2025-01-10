from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms


def Loading_Dataset(data_dir:str):
    
    train=CIFAR10(root=data_dir,
                    train=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,)),
                                                    transforms.Resize((64,64))]),
                    download=True)

    test=CIFAR10(root=data_dir,
                 train=False,
                 transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,),(0.5,)),
                                                transforms.Resize((64,64))]),
                 download=True)


    return train,test


def dataloader(train_dataset,test_dataset,batch_size):
    
    train=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    
    
    return train,test
    
    



