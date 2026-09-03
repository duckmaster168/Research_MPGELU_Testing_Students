import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataTransforms:
    '''Data transformation utility handling image augmentation and dataset statistics normalization[cite: 4].'''
    def __init__(self, dataset: str, use_augmentation=True, use_stats=False):
        self.dataset = dataset[cite: 4]
        self.use_augmentation = use_augmentation[cite: 4]
        self.use_stats = use_stats[cite: 4]
        
    def get_train_transform(self):
        if not self.use_augmentation:[cite: 4]
            return self.get_test_transform()[cite: 4]
        
        transforms_list = [
            transforms.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.3, hue=0.2),[cite: 4]
            transforms.RandomCrop(32, padding=4),[cite: 4]
            transforms.RandomRotation(10),[cite: 4]
            transforms.RandomHorizontalFlip(p=0.5),[cite: 4]
            transforms.RandomVerticalFlip(p=0.2),[cite: 4]
            transforms.ToTensor()[cite: 4]
        ]
        
        if self.use_stats and self.dataset.lower() == 'cifar10':[cite: 4]
            transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))[cite: 4]
            
        return transforms.Compose(transforms_list)[cite: 4]
    
    def get_test_transform(self):
        transforms_list = [
            transforms.ToTensor(),[cite: 4]
        ]
        
        if self.use_stats and self.dataset.lower() == 'cifar10':[cite: 4]
            transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))[cite: 4]
            
        return transforms.Compose(transforms_list)[cite: 4]
