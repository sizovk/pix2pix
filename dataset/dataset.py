import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T


class Edges2Handbags(Dataset):
    
    def __init__(self, data_folder="data/edges2handbags/", mode="train"):
        super().__init__()
        data_path = os.path.join(data_folder, mode)
        self.image_paths = [os.path.join(data_path, image) for image in sorted(os.listdir(data_path))]
        self.resize_transform = T.Resize((256, 512))
        self.grayscale_transform = T.Grayscale(1)
    
    def __getitem__(self, index):
        image = read_image(self.image_paths[index]) / 255
        image = self.resize_transform(image)
        edge = self.grayscale_transform(image[:,:,:256])
        handbag = image[:,:,256:]
        return edge, handbag
    
    def __len__(self):
        return len(self.image_paths)


class Edges2Shoes(Dataset):
    
    def __init__(self, data_folder="data/edges2shoes/", mode="train"):
        super().__init__()
        data_path = os.path.join(data_folder, mode)
        self.image_paths = [os.path.join(data_path, image) for image in sorted(os.listdir(data_path))]
        self.resize_transform = T.Resize((256, 512))
        self.grayscale_transform = T.Grayscale(1)
    
    def __getitem__(self, index):
        image = read_image(self.image_paths[index]) / 255
        image = self.resize_transform(image)
        edge = self.grayscale_transform(image[:,:,:256])
        handbag = image[:,:,256:]
        return edge, handbag
    
    def __len__(self):
        return len(self.image_paths)

