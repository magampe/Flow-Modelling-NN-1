# define packages and machine information ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from  torchvision.transforms import ToTensor, v2

import pandas as pd
import rasterio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42069)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# define context dataset  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GeoContextData(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = torch.tensor(df['Discharge (Mean)'].values).unsqueeze(-1)
        img_count = len(self.img_names)
        self.all_data = torch.empty((img_count,3,224,224))

        self.transform = transform

        # Load data into memory
        for i,n in enumerate(self.img_names):
            
            with rasterio.open(os.path.join(self.img_dir + n), 'r') as src:
                img = src.read() 
                img = ToTensor()(img)
                data = F.normalize(img, p = 2.0, dim = 0)
                # Pad data to create square images
                data = F.pad(data, (0,0,0,0,0,495))
                # get dimensions in right spots
                data = torch.permute(data, (1,2,0))
                # Limit highly negative values
                data = torch.clamp(data,min=-1,max=None)
                # reshape to resnet standard size
                data = v2.functional.resize(data, (224,224))
                # Add data to full tensor
                self.all_data[i] = data

        # get dimensions in right spots
        #print(self.all_data.shape)

        # self.all_data = torch.permute(self.all_data, (1,2,0))

    def __getitem__(self, idx):
        
        # Get image
        data = self.all_data[idx]

        # Get flow rate
        discharge = self.y[idx]

        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        return data, discharge

    def __len__(self):
        return self.y.shape[0]

# define encoder/decoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(3, 64, 9, 1, 0), # input dim: (3, 224, 224); next dim: (64, 215, 215)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),

            # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),

            # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            
            # Final encoded image size (batch first): torch.Size([5, 128, 106, 106])
            
            # using a strided convultion to compress instead of maxpool as
            # maxpool doesn't work in nn.sequential (need to save
            # another output from it to feed to unpool in decoder).
        )

        self.decoder = nn.Sequential(
            nn.Upsample((128,128),  mode='nearest'),
            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Upsample((64,64), mode='nearest'),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # 64, 358, 358
            nn.ReLU(True),
            
            nn.Upsample((224,224), mode='nearest'),
            nn.Conv2d(64, 3, 9, stride=1, padding=4),
            nn.Tanh(),  
        )

    def forward(self,x):
        encoded = self.encoder(x).to(device)
        decoded = self.decoder(encoded).to(device)
        out = []
        out.append(encoded)
        out.append(decoded)
        return out
    
# actually running this as standalone
    
def main():

    # define transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    transform = None
        
    # define paths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    train_dataset = GeoContextData(csv_path='Data/Q2_2022/target/Q2_2022_target_train.csv',
                            img_dir='Data/Q2_2022/STACK/train/',
                            transform=transform)

    validation_dataset = GeoContextData(csv_path='Data/Q2_2022/target/Q2_2022_target_valid.csv',
                            img_dir='Data/Q2_2022/STACK/validation/',
                            transform=transform)

    # test_dataset = GeoContextData(csv_path='Data/Q2_2022/target/Q2_2022_target_test.csv',
    #                           img_dir='Dat/Q2_2022/STACK/test/',
    #                           transform=transform)

    # dataloader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=5,
                            drop_last=True,
                            shuffle=True, # want to shuffle the dataset
                            num_workers=0)

    valid_loader = DataLoader(dataset=validation_dataset,
                            batch_size=5,
                            drop_last=True,
                            shuffle=True, # want to shuffle the dataset
                            num_workers=0)

    # test_loader = DataLoader(dataset=test_dataset,
    #                           batch_size=5,
    #                           drop_last=True,
    #                           shuffle=True, # want to shuffle the dataset
    #                           num_workers=0)

    
        
    # model parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model  = CNNAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 20
    outputs=[]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # data_loader = train_loader
    for epoch in range(num_epochs):
        for(img,_) in train_loader:
            out = model(img)
            encode = out[0].to(device)
            recon = out[1].to(device)
            loss = criterion(recon,img)
            # print(type(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item():4f}')
        outputs.append((epoch,img,recon))

if __name__ == "__main__":
    main()