from torch.utils.data import Dataset, DataLoader ,random_split,Subset
from torchvision import transforms, models,datasets
# Custom dataset class for splitting the data into train and validation sets
class root_dataset(Dataset):
    def __init__(self,path):
        self.dataset1=datasets.ImageFolder(root=(path+'/train'))    # Load the dataset from the provided path
        l1=int(len(self.dataset1)*0.8)
        train_dataset,val_dataset=random_split(self.dataset1, [int(len(self.dataset1)*0.8),len(self.dataset1)-l1])   # Split the dataset into train and validation sets
        #print(len(train_dataset),len(val_dataset))
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
    def get_train_data(self):
        return self.train_dataset
    def get_val_data(self):
        return self.val_dataset

class inaturalist_train(Dataset):   # Custom dataset class for training data
    def __init__(self,train_data):
        self.target_size=(3,224,224)    # Define target image size
        #dataset1=datasets.ImageFolder(root='(path+'/train')')
        self.dataset=train_data
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size[1:]),   # Resize images to target size
            transforms.ToTensor()
        ])#self.target_size = target_size

    def __getitem__(self,idx):
        image,label=self.dataset[idx]
        image=self.transform(image)
        return image,label
    def __len__(self):
        return len(self.dataset)

class inaturalist_val(Dataset):         # Custom dataset class for validation data
    def __init__(self,val_data):
        self.target_size=(3,224,224)    # Define target image size
        #dataset1=datasets.ImageFolder(root='/kaggle/input/neurolist/inaturalist_12K/train')
        self.dataset=val_data
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size[1:]),      # Resize images to target size
            transforms.ToTensor()
        ])#self.target_size = target_size
    def __getitem__(self,idx):         
        image,label=self.dataset[idx]
        image=self.transform(image)
        return image,label
    def __len__(self):
        return len(self.dataset)

class inaturalist_test(Dataset):     # Custom dataset class for test data
    def __init__(self,path):
        #C:/Users/USER/Downloads/nature_12K/inaturalist_12K
        self.target_size=(3,224,224)      # Define target image size
        self.dataset=datasets.ImageFolder(root= path+'/val')     # Load the test dataset from the provided path
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size[1:]),  # Resize images to target size
            transforms.ToTensor()
        ])#self.target_size = target_size   # Resize images to target size
    def __getitem__(self,idx):
        image,label=self.dataset[idx]
        image=self.transform(image)
        return image,label
    def __len__(self):
        return len(self.dataset)
