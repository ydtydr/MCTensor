# %%
import torch
import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random, os, math
import matplotlib.pyplot as plt
from MCTensor.MCModule import *
from MCTensor.MCOptim import *
from tqdm import tqdm
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

###############################################################################
d16 = torch.float16
d32 = torch.float32
d64 = torch.float64
cpu = torch.device("cpu")
gpu = torch.device(type='cuda', index=0)
# set device
device = gpu

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2,
                 dtype=d16, device=device,
                 fc1_w=None, fc2_w=None, fc3_w=None):
        super(MLP,self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden1, 
                             dtype=dtype, device=device, bias=False)
        self.fc2 = nn.Linear(hidden1,hidden2, 
                             dtype=dtype, device=device, bias=False)
        self.fc3 = nn.Linear(hidden2, 10, 
                             dtype=dtype, device=device, bias=False)
        self.droput = nn.Dropout(0.2)
        
        if fc1_w is not None:             
            self.fc1.weight.data.copy_(fc1_w.data.to(dtype))
        if fc2_w is not None:
            self.fc2.weight.data.copy_(fc2_w.data.to(dtype))
        if fc3_w is not None:
            self.fc3.weight.data.copy_(fc3_w.data.to(dtype))
        
    def forward(self,x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        # return x
    
    
class MCMLP(MCModule):
    def __init__(self,input_dim, hidden1, hidden2, nc=2, dtype=d16, device=device,
                 fc1_w=None, fc2_w=None, fc3_w=None):
        super(MCMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = MCLinear(input_dim, hidden1, nc=nc, bias=False,
                            dtype=dtype, device=device, _weight=fc1_w)
        self.fc2 = MCLinear(hidden1, hidden2, nc=nc, bias=False,
                            dtype=dtype, device=device, _weight=fc2_w)
        self.fc3 = MCLinear(hidden2, 10, nc=nc, bias=False,
                            dtype=dtype, device=device, _weight=fc3_w)
        self.droput = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = x.tensor.sum(-1)
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = x.tensor.sum(-1)
        x = self.droput(x)
        x = self.fc3(x)
        x = x.tensor.sum(-1)
        x = F.log_softmax(x, dim=1)        
        return x 

# %%
def get_dataloader(dtype=d32, batch_size=32):
    seed_everything()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.ConvertImageDtype(dtype)])
    dataset = datasets.MNIST(root = 'data', train = True, 
                                download = True, transform = transform)
    
    indices = np.arange(len(dataset))
    train_indices, _ = train_test_split(indices, train_size=1000*10, random_state=42)
    entire_data = Subset(dataset, train_indices)
    
    idx = np.arange(len(entire_data))
    t_idx, v_idx = train_test_split(idx, test_size=0.2)
    train_data = Subset(entire_data, t_idx)
    valid_data = Subset(entire_data, v_idx)
    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=False)
    
    seed_everything()
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle=False)
    return train_loader, valid_loader

# %%
def init_weights(input_dim, hidden1, hidden2, seed=1234, dtype=d32, device=device):
    seed_everything(seed=seed)
    input_dim = input_dim
    std1 = 1./math.sqrt(hidden1)
    fc1_w = torch.zeros(input_dim, hidden1)
    fc1_w.data.uniform_(-std1, std1).to(dtype).to(device)
    std2 = 1./math.sqrt(hidden2)
    fc2_w = torch.zeros(hidden1, hidden2)
    fc2_w.data.uniform_(-std2, std2).to(dtype).to(device)
    fc3_w = torch.zeros(hidden2, 10)
    fc3_w.data.uniform_(-1e-1, 1e-1).to(dtype).to(device)
    return fc1_w, fc2_w, fc3_w


# %%
def train_torch(dtype, epochs=5, hidden1=128, hidden2=128, lr=0.001,
                device=device, init_seed=1234, B=32, input_dim=28*28):
    train_loader, valid_loader = get_dataloader(dtype=dtype, batch_size=B)
    LOSS, LOSS_test, acc, acc_test = [], [], [], []
    fc1_w, fc2_w, fc3_w = init_weights(input_dim, hidden1, hidden2, seed=init_seed, dtype=dtype)
    model = MLP(input_dim, hidden1, hidden2,
                dtype=dtype, device=device,
                fc1_w=fc1_w.t(), fc2_w=fc2_w.t(), fc3_w=fc3_w.t())
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
 
    def get_ncorrect(output, target):
        pred = output.data.max(1)[1]
        n_correct = (pred == target).sum()
        return n_correct
    
    def get_acc(output, target):
        pred = output.data.max(1)[1]
        n_correct = (pred == target).sum()
        assert len(pred) == len(target)
        acc = n_correct / len(target)
        return 100 * acc.item()

    
    for epoch in tqdm(range(epochs)):
        # trainloss=0
        # validloss=0
        # ncorrect_train = 0
        # ncorrect_test = 0
        etrain = []
        etest = []
        eltrain = []
        eltest = []
        kt = 0
        kv = 0
        model.train()
        # with tqdm(train_loader, unit="batch") as tepoch:
        for X_train, y_train in train_loader:
            kt += 1
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(X_train)
            # print(output.size(), output.device)
            # print(y_train.size(), y_train.device)
            loss = criterion(output, y_train)

            # trainloss += loss.data.item()
            # ncorrect_train += get_ncorrect(output, y_train)
            loss.backward()
            optimizer.step()
            eltrain.append(loss.data.item())
            etrain.append(get_acc(output, y_train))
            # if kt >= 100:
            #     break
                
        eltrain = np.array(eltrain)
        etrain = np.array(etrain) 
        acc.append(np.mean(etrain))
        LOSS.append(np.mean(eltrain))
        
        
        model.eval()
        # with tqdm(valid_loader, unit="batch") as ttepoch:
        with torch.no_grad():
            for X_test, y_test in valid_loader:
                kv += 1
                X_test, y_test = X_test.to(device), y_test.to(device)
                output_test = model(X_test)
                loss_test = criterion(output_test, y_test)
                # validloss += loss_test.data.item()
                # ncorrect_test += get_ncorrect(output_test, y_test)
                eltest.append(loss_test.data.item())
                etest.append(get_acc(output_test, y_test))
                if kv >= 3:
                    break
                    
            eltest = np.array(eltest)
            etest = np.array(etest)
            acc_test.append(np.mean(etest))
            LOSS_test.append(np.mean(eltest))
            # LOSS_test.append(validloss / len(valid_loader))

            

    return model, LOSS, acc, LOSS_test, acc_test


# %%
lr=5e-3
epochs=70

_, LOSS16, acc16, LOSS_test16, acc_test16 = train_torch(d16,epochs=epochs, lr=lr)
_, LOSS32, acc32, LOSS_test32, acc_test32 = train_torch(d32,epochs=epochs, lr=lr)
_, LOSS64, acc64, LOSS_test64, acc_test64 = train_torch(d64,epochs=epochs, lr=lr)

print(f"NLL_the train loss for d16: {LOSS16[-1]}, test acc for d16 {acc_test16[-1]}")
print(f"NLL_the train loss for d32: {LOSS32[-1]}, test acc for d32 {acc_test32[-1]}")
print(f"NLL_the train loss for d64: {LOSS64[-1]}, test acc for d64 {acc_test64[-1]}")

np.save('LOSS32.npy', np.array(LOSS32))
np.save('acc32.npy', np.array(acc32))
np.save('LOSStest32.npy', np.array(LOSS_test32))
np.save('acctest32.npy', np.array(acc_test32))

np.save('LOSS16.npy', np.array(LOSS16))
np.save('acc16.npy', np.array(acc16))
np.save('LOSStest16.npy', np.array(LOSS_test16))
np.save('acctest16.npy', np.array(acc_test16))

np.save('LOSS64.npy', np.array(LOSS64))
np.save('acc64.npy', np.array(acc64))
np.save('LOSStest64.npy', np.array(LOSS_test64))
np.save('acctest64.npy', np.array(acc_test64))

# %%
def train_MC(dtype,nc=2, epochs=5, hidden1=128, hidden2=128, lr=1e-2,
             input_dim = 28*28, device=device, init_seed=1234, B=32):
    train_loader, valid_loader = get_dataloader(dtype=dtype, batch_size=B)
    LOSS, LOSS_test, acc, acc_test = [], [], [], []
    fc1_w, fc2_w, fc3_w = init_weights(input_dim, hidden1, hidden2, seed=init_seed, dtype=dtype)
    model = MCMLP(input_dim, hidden1, hidden2, nc=nc,
                dtype=dtype, device=device,
                fc1_w=fc1_w.t(), fc2_w=fc2_w.t(), fc3_w=fc3_w.t())
    criterion = torch.nn.NLLLoss()
    optimizer = MCSGD(model.parameters(), lr=lr, momentum=0.9)
 
    def get_ncorrect(output, target):
        pred = output.data.max(1)[1]
        n_correct = pred.eq(target.data).sum()
        return n_correct

    def get_acc(output, target):
        pred = output.data.max(1)[1]
        n_correct = (pred == target).sum()
        assert len(pred) == len(target)
        acc = n_correct / len(target)
        return 100 * acc.item()

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    for epoch in tqdm(range(epochs)):
        etrain = []
        etest = []
        eltrain = []
        eltest = []
        kt = 0
        kv = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for X_train, y_train in tepoch:
            # for X_train, y_train in train_loader:
                kt += 1
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)

                # trainloss += loss.data.item()
                # ncorrect_train += get_ncorrect(output, y_train)
                loss.backward()
                optimizer.step()
                eltrain.append(loss.data.item())
                etrain.append(get_acc(output, y_train))
                # if kt >= 10:
                #     break
                
            eltrain = np.array(eltrain)
            etrain = np.array(etrain) 
            acc.append(np.mean(etrain))
            LOSS.append(np.mean(eltrain))
        
        
        model.eval()
        with tqdm(valid_loader, unit="batch") as ttepoch:
            with torch.no_grad():
                for X_test, y_test in ttepoch:
                # for X_test, y_test in valid_loader:
                    kv += 1
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    output_test = model(X_test)
                    loss_test = criterion(output_test, y_test)
                    # validloss += loss_test.data.item()
                    # ncorrect_test += get_ncorrect(output_test, y_test)
                    eltest.append(loss_test.data.item())
                    etest.append(get_acc(output_test, y_test))
                    if kv >= 3:
                        break
                        
                eltest = np.array(eltest)
                etest = np.array(etest)
                acc_test.append(np.mean(etest))
                LOSS_test.append(np.mean(eltest))
            # LOSS_test.append(validloss / len(valid_loader))  
    return model, LOSS, acc, LOSS_test, acc_test

# %%
_, LOSS16_nc1, acc16_nc1, LOSS16_test_nc1, acc16_test_nc1 = train_MC(d16,nc=1, epochs=epochs,lr=lr)
print(f"for nc=1, the train loss: {LOSS16_nc1[-1]}, and the test acc:{acc16_test_nc1[-1]}")
np.save('LOSS16nc1.npy', np.array(LOSS16_nc1))
np.save('acc16nc1.npy', np.array(acc16_nc1))
np.save('LOSStest16nc1.npy', np.array(LOSS16_nc1))
np.save('acctest16nc1.npy', np.array(acc16_test_nc1))


_, LOSS16_nc2, acc16_nc2, LOSS16_test_nc2, acc16_test_nc2 = train_MC(d16,nc=2, epochs=epochs,lr=lr)
print(f"for nc=2, the train loss: {LOSS16_nc2[-1]}, and the test acc:{acc16_test_nc2[-1]}")
np.save('LOSS16nc2.npy', np.array(LOSS16_nc2))
np.save('acc16nc2.npy', np.array(acc16_nc2))
np.save('LOSStest16nc2.npy', np.array(LOSS16_nc2))
np.save('acctest16nc2.npy', np.array(acc16_test_nc2))


_, LOSS16_nc3, acc16_nc3, LOSS16_test_nc3, acc16_test_nc3 = train_MC(d16,nc=3, epochs=epochs,lr=lr)
print(f"for nc=3, the train loss: {LOSS16_nc3[-1]}, and the test acc:{acc16_test_nc3[-1]}")
np.save('LOSS16nc3.npy', np.array(LOSS16_nc3))
np.save('acc16nc3.npy', np.array(acc16_nc3))
np.save('LOSStest16nc3.npy', np.array(LOSS16_nc3))
np.save('acctest16nc3.npy', np.array(acc16_test_nc3))
