# define packages and machine information ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from  torchvision.transforms import ToTensor, Compose, Normalize, Resize, v2
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np 
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from encoder import CNNAutoencoder
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.manual_seed(42069)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)

# import data and pre-process ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: load up encoded model
def encoder():
    model = CNNAutoencoder().to(device)
    params = torch.load("./CNN2.pt", map_location = torch.device(device))
    model.load_state_dict(params)
    
    with rasterio.open("Data/2016_2021/03539600_DEM.tif", 'r') as src:
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
        data = v2.functional.resize(data, (224,224)).unsqueeze(0).to(torch.float64)
        data.float()
    
    _test = model(data)
    return _test[0]
ho = encoder()


# h0 = ho.flatten()
# h0 = self.encolin(h0)
# h0 = h0.repeat(lstm_layers, batch_size,  1)
# c0 = h0

# Define dataloader for loading images

class SeqGeoImageDataset(Dataset):

    def __init__(self, csv_path, img_dir, seq_len=1, forecast=1, transform=None, scale=(1.0,1.0,1.0)):
    
        # save configuration info
        
        self.transform = transform
        self.seq_len = seq_len
        self.forecast = forecast
        self.scale = scale

        # load target data into memory
        
        df = pd.read_csv(csv_path)
        self.y = torch.tensor(df['Discharge (Mean)'].values).unsqueeze(-1)
        
        #self.magnitude = self.y.norm(p=2, dim=0, keepdim=Flase)
        self.y = self.y / self.scale[0]

        # load image data into memory
        
        self.img_dir = img_dir
        self.img_names = df['File Name']
        img_count = len(self.img_names)
        self.all_data = torch.empty((img_count,2,8,8))
        
        for i,n in enumerate(self.img_names):
            
            with rasterio.open(os.path.join(self.img_dir + n), 'r') as src:
                img = src.read() 
                img = ToTensor()(img)

            img = torch.permute(img, (1,2,0))
            img[0] = torch.clamp(img[0],min=-99,max=None)
            img[1] = torch.clamp(img[1],min=-1,max=None)
            self.all_data[i] = img
        
        self.all_data[:,0] = self.all_data[:,0] / self.scale[1]
        self.all_data[:,1] = self.all_data[:,1] / self.scale[2]

    def __getitem__(self, idx):
        

        # load the images sequences
        datas = self.all_data[idx:idx+self.seq_len]

        # get flow rates
        discharge = self.y[idx + self.seq_len + self.forecast]

        # apply transforms
        if self.transform:
            datas = self.transform(datas)

        return datas, discharge

    def __len__(self):
        return self.y.shape[0] - self.seq_len - self.forecast

# define dataloader for loading prior flow rates

class SeqFlowDataset(Dataset):

    def __init__(self, csv_path, seq_len=1, forecast=1, transform=None, scale=(1.0,1.0,1.0)):
    
        df = pd.read_csv(csv_path)
        self.y = df['Discharge (Mean)']
        self.transform = transform
        self.seq_len = seq_len
        self.forecast = forecast
        self.scale = scale

        # load target data into memory
        
        df = pd.read_csv(csv_path)
        self.y = torch.tensor(df['Discharge (Mean)'].values).unsqueeze(-1)
        self.y = self.y / self.scale[0]
    
    def __getitem__(self, idx):

        # get prior flow rates

        datas = self.y[idx:idx+self.seq_len]

        # get flow rates

        discharge = self.y[idx + self.seq_len + self.forecast]

        # apply transforms
        
        if self.transform:
            datas = self.transform(datas)

        return datas, discharge

    def __len__(self):
        return self.y.shape[0] - self.seq_len - self.forecast

image_transform = None
flow_transform = None

# plotting and logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pred_train_logger = []
pred_valid_logger = []
pred_test_logger = []

y_train_logger = []
y_valid_logger = []
y_test_logger = []

loss_train_logger = []
loss_val_logger = []
loss_test_logger = []

err_train_logger = []
err_val_logger = []
err_test_logger = []

def plot_loss(loss_train_logger, loss_val_logger):
    fig, ax = plt.subplots()
    ax.plot(loss_train_logger[4:], label="Training Loss")
    ax.plot(loss_val_logger[4:], label="Validation Loss")
    #ax.set_ylim([0, 0.03])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid()
    fig.savefig("./Figures/Loss.png")
    plt.close(fig)

def plot_error(err_train_logger, err_val_logger):
    fig, ax = plt.subplots()
    ax.plot(err_train_logger[4:], label="Training Error")
    ax.plot(err_val_logger[4:], label="Validation Error")
    #ax.set_ylim([0, 10])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Relative Error")
    ax.set_title("Relative Error Curves")
    ax.legend()
    ax.grid()
    fig.savefig("./Figures/Error.png")
    plt.close(fig)

def plot_train_preds(tpreds, ty):
    fig, ax = plt.subplots()
    ax.plot(tpreds, label="Predicted Q")
    ax.plot(ty, label="Q")
    #ax.set_ylim([0, 0.03])
    ax.set_xlabel("Time")
    ax.set_ylabel("Measure")
    ax.set_title("Training Predictions Vs. Actuals")
    ax.legend()
    ax.grid()
    fig.savefig("./Figures/Preds_T.png")
    plt.close(fig)

def plot_valid_preds(vpreds, vy):
    fig, ax = plt.subplots()
    ax.plot(vpreds, label="Predicted Q")
    ax.plot(vy, label="Q")
    #ax.set_ylim([0, 0.03])
    ax.set_xlabel("Time")
    ax.set_ylabel("Measure")
    ax.set_title("Validation Predictions Vs. Actuals")
    ax.legend()
    ax.grid()
    fig.savefig("./Figures/Preds_V.png")
    plt.close(fig)

def plot_test_preds(testpreds, testy):
    fig, ax = plt.subplots()
    ax.plot(testpreds, label="Predicted Q")
    ax.plot(testy, label="Q")
    #ax.set_ylim([0, 0.03])
    ax.set_xlabel("Time")
    ax.set_ylabel("Measure")
    ax.set_title("Test Predictions Vs. Actuals")
    ax.legend()
    ax.grid()
    fig.savefig("./Figures/Test Predictions.png")
    plt.close(fig)


# define network architecture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Build an LSTM

class LSTMRegressor(torch.nn.Module):
        def __init__(
                self,
                input_dim,
                embedding_dim,
                hidden_dim,
                dropout,
                lstm_layers
                ):
             
            super(LSTMRegressor, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim

            # batchnorm layer
            self.normie = nn.BatchNorm3d(2)

            # something to squeeze down images
            self.linear_preprocess = nn.Sequential(
                nn.Linear(self.input_dim, self.embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).float()

            # the lstm itself
            self.rnn = nn.LSTM(
                input_size = self.embedding_dim + (1 if mode=="both" else 0),
                hidden_size = self.hidden_dim,
                num_layers = lstm_layers,
                batch_first = True,
                dropout = dropout,
                ).float()
            
            # linear layer to convert hidden space to a flow rate?
            self.to_flow = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim).float(),
                nn.LeakyReLU().float(),
                nn.Linear(hidden_dim, 1).float(),
            ).float()

            self.encolin = torch.nn.Linear(106*106*128, self.hidden_dim).float()
            self.ReLU = nn.ReLU().float()

        def forward(self, data):

            if mode == "images":
                image = data
                embed = image.flatten(start_dim=-3,end_dim=-1)
                embed = self.linear_preprocess(embed)
                embed = embed.float()
            
            elif mode == "both":
                image, flow = data
                flow *= scale[0]
                embed = image.flatten(start_dim=-3,end_dim=-1)
                # .to(torch.float32)
                embed = self.linear_preprocess(embed)
                embed = torch.cat((embed,flow),dim=2).float()
            
            elif mode == "flows":
                embed = data.float()
            
            else:
                print("Something went wrong")
                exit()
            h0 = ho.flatten()
            h0 = self.encolin(h0)
            h0 = self.ReLU(h0)
            h0 = h0.repeat(lstm_layers, batch_size,  1)
            c0 = h0
            lstm_out, (h, c) = self.rnn(embed, (h0,c0))
            lstm_out = lstm_out[:, -1, :]
            Q = self.to_flow(lstm_out)

            return Q

# define hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# custom loss **************************

def rel_error_loss(pred, y):
    return (torch.abs(pred - y) / (y + 1e-5)).mean()

def rel_square_error_loss(pred, y):
    return (torch.square((pred - y) / (y + 1e-5) + 1) - 1).mean()

#  generic  params *********************

epochs = 300
dropout = 0.0
# loss_func = nn.L1Loss()
# loss_func = nn.MSELoss()
# loss_func = nn.HuberLoss()
loss_func = rel_error_loss
# loss_func = rel_square_error_loss
batch_size = 128

# LSTM params **************************

embedding_size = 128
hidden_dim =  128
lstm_layers = 2

# optimization params ******************

learning_rate = 1e-4 #1e-4 good for images, 5e-6 good for both sets, 
amsgrad=True
weight_decay=1e-5

gamma=0.9900

# dataloader params ********************

# you need to delete the *.pt in /Data before changing these
forecast = 1 # how far into future to predict
seq_len = 150
mode = "both" # "images", "flows", "both"


# TODO: load up encoded model
def encoder():
    model = CNNAutoencoder(hidden_dim=hidden_dim).to(device)
    params = torch.load("./CNN2.pt", map_location = torch.device(device))
    model.load_state_dict(params)
    
    with rasterio.open("Data/2016_2021/03539600_DEM.tif", 'r') as src:
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
        data = v2.functional.resize(data, (224,224)).unsqueeze(0).to(torch.float64)
        data.float()
    
    _test = model(data)
    return _test[0]
ho = encoder()

# load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

save_file = "Data/sequence.pt"
# scale = (9000,99,145.14) # <-- magnitude scale
scale = (9000.0, 1.0, 1.0)

if os.path.exists(save_file) == False:
    image_train_dataset = SeqGeoImageDataset(csv_path='Data/2016_2021/lstm/train.csv',
                            img_dir='Data/2016_2021/STACK/train/',
                            transform=image_transform,
                            seq_len=seq_len,
                            forecast=forecast,
                            scale=scale
                            )
    flow_train_dataset = SeqFlowDataset(csv_path='Data/2016_2021/lstm/train.csv',
                            transform=flow_transform,
                            seq_len=seq_len,
                            forecast=forecast,
                            scale=scale
                            )
    image_validation_dataset = SeqGeoImageDataset(csv_path='Data/2016_2021/lstm/valid.csv',
                            img_dir='Data/2016_2021/STACK/validation/',
                            transform=image_transform,
                            seq_len=seq_len,
                            forecast=forecast,
                            scale=scale
                            )
    flow_validation_dataset = SeqFlowDataset(csv_path='Data/2016_2021/lstm/valid.csv',
                            transform=flow_transform,
                            seq_len=seq_len,
                            forecast=forecast,
                            scale=scale
                            )
    flow_test_set = SeqFlowDataset(csv_path='Data/2022_2023/TARGET/lstm_test.csv',
                                    transform=flow_transform,
                                    seq_len=seq_len,
                                    forecast=forecast,
                                    scale=scale)
    image_test_set = SeqGeoImageDataset(csv_path='Data/2022_2023/TARGET/lstm_test.csv',
                            img_dir='Data/2022_2023/STACK/',
                            transform=image_transform,
                            seq_len=seq_len,
                            forecast=forecast,
                            scale=scale)

    all_sets = (
        image_train_dataset, 
        flow_train_dataset,
        image_validation_dataset,
        flow_validation_dataset,
        image_test_set,
        flow_test_set
    )
    torch.save(all_sets, save_file)

else:
    image_train_dataset, \
        flow_train_dataset, \
            image_validation_dataset, \
                flow_validation_dataset, \
                    image_test_set, \
                        flow_test_set = torch.load(save_file)

# make loaders

combo_train_set = torch.utils.data.StackDataset(image_train_dataset, flow_train_dataset)
combo_valid_set = torch.utils.data.StackDataset(image_validation_dataset, flow_validation_dataset)
combo_test_set = torch.utils.data.StackDataset(image_test_set, flow_test_set)

if mode == "images":
    train = image_train_dataset
    valid = image_validation_dataset
    test = image_test_set
    input_size = 128
elif mode == "both":
    train = combo_train_set
    valid = combo_valid_set
    test = combo_test_set
    input_size = 128
elif mode == "flows":
    train = flow_train_dataset
    valid = flow_validation_dataset
    test = flow_test_set
    input_size = 1
else:
    print("Wrong hyperparameter used for 'mode'. Use 'images', 'both',or 'flows'.")
    exit()

train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0)

valid_loader = DataLoader(dataset=valid,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0)

test_loader = DataLoader(dataset=test,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0)

# define optimizer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#input_size = 128 # <----- this needs to change when moving to a real encoder

model = LSTMRegressor(
     input_dim=input_size,
     embedding_dim=embedding_size, 
     hidden_dim=hidden_dim,
     dropout=dropout,
     lstm_layers=lstm_layers
     )

model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    amsgrad=amsgrad,
    weight_decay=weight_decay
    )

scheduler = ExponentialLR(optimizer, gamma=gamma)

# define training function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def trainer(
          dataloader,
          model,
          loss_func,
          optimizer
    ):

    model.train() # <--- just puts model into training mode
    running_loss = 0.0
    running_err = 0.0
    batch_count = len(dataloader)

    #for X, y in dataloader:
    for left, right in dataloader:

        if type(left) == list:
            X1, y = left
            X2, _ = right
            X = (X1.to(device), X2.to(device))
            y = y.to(device)
        else:
            X = left.to(device)
            y = right.to(device)

        preds = model(X).to(device)
        loss = loss_func(preds, y).to(device)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()
            running_err += (torch.abs(preds - y) / y).mean().item()
        



    print(f"Predictions: {preds[0:5].T * scale[0]}")
    print(f"    Actuals: {y[0:5].T * scale[0]}")

    running_loss = running_loss / batch_count
    running_err = running_err / batch_count
    print(f"Average Training Loss:             {running_loss:>8f}")
    print(f"Average Training Relative Error:   {running_err:>8f}\n")
    loss_train_logger.append(running_loss)
    err_train_logger.append(running_err)
    return preds, y
    
    

# define validation function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def validator(
        dataloader,
        model,
        loss_func
    ):

    model.eval() # <--- put model into evaluation mode
    batch_count = len(dataloader)
    running_loss = 0.0
    running_err = 0.0

    with torch.no_grad():
        
        for left, right in dataloader:
            if type(left) == list:
                X1, y = left
                X2, _ = right
                X = (X1.to(device), X2.to(device))
                y = y.to(device)
            else:
                X = left.to(device)
                y = right.to(device)

            preds = model(X).to(device)
            running_loss += loss_func(preds, y).item()
            running_err += (torch.abs(preds - y) / y).mean().item()

    running_loss = running_loss / batch_count
    running_err = running_err / batch_count
    print(f"Average Validation Loss:           {running_loss:>8f}")
    print(f"Average Validation Relative Error: {running_err:>8f} \n")
    loss_val_logger.append(running_loss)
    err_val_logger.append(running_err)
    return preds, y

# define testing function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tester(
        dataloader,
        model,
        loss_func
    ):

    model.eval() # <--- put model into evaluation mode
    batch_count = len(dataloader)
    running_loss = 0.0
    running_err = 0.0

    with torch.no_grad():
        
        for left, right in dataloader:
            if type(left) == list:
                X1, y = left
                X2, _ = right
                X = (X1.to(device), X2.to(device))
                y = y.to(device)
            else:
                X = left.to(device)
                y = right.to(device)

            preds = model(X).to(device)
            running_loss += loss_func(preds, y).item()
            running_err += (torch.abs(preds - y) / y).mean().item()

    running_loss = running_loss / batch_count
    running_err = running_err / batch_count
    print(f"Average Test Loss:           {running_loss:>8f}")
    print(f"Average Test Relative Error: {running_err:>8f} \n")
    loss_test_logger.append(running_loss)
    err_test_logger.append(running_err)
    return preds, y

# run the jewels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# LSTM

start = time.time()
for ep in range(epochs):

    print(f" EPOCH {ep+1} ".center(20,"*"))
    trainer(
        train_loader,
        model,
        loss_func,
        optimizer
    )
    scheduler.step()
    validator(
        valid_loader,
        model,
        loss_func
    )
    tester(
        test_loader,
        model,
        loss_func
    )

    if ep == 262:
        tpreds, ty = trainer(train_loader,model,loss_func,optimizer)
        tpreds = tpreds.cpu().detach().numpy()
        ty = ty.cpu().detach().numpy()

        vpreds, vy = validator(valid_loader,model,loss_func)
        vpreds = vpreds.cpu().detach().numpy()
        vy = vy.cpu().detach().numpy()

        testpreds, testy = tester(test_loader,model,loss_func)
        testpreds = testpreds.cpu().detach().numpy()
        testy = testy.cpu().detach().numpy()
        
    

print("\nModel finished training and testing in:")
print(f"{time.time() - start} seconds.\n")

min_train_err = min(err_train_logger)
min_train_epo = err_train_logger.index(min_train_err) + 1

min_valid_err = min(err_val_logger)
min_valid_epo = err_val_logger.index(min_valid_err) + 1

min_test_err = min(err_test_logger)
min_test_epo = err_test_logger.index(min_test_err) + 1

print(f"Best Training Error: [{min_train_err}] @ Epoch [{min_train_epo}]")
print(f"Best Validation Error: [{min_valid_err}] @ Epoch [{min_valid_epo}]")
print(f"Best Test Error: [{min_test_err}] @ Epoch [{min_test_epo}]")

plot_loss(loss_train_logger, loss_val_logger)
plot_error(err_train_logger, err_val_logger)
plot_train_preds(tpreds,ty)
plot_valid_preds(vpreds, vy)
plot_test_preds(testpreds, testy)




