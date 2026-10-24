import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from actnorm import ActNorm
from actnorm import ActNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class FunctionNet(nn.Module):
    def __init__(self, channels_in, channels_out, device, init_last_zeros=False):
        super(FunctionNet, self).__init__()
        self.channels_hidden = 256
        self.conv1 = nn.Conv2d(channels_in, self.channels_hidden, kernel_size=(3,3), stride=1, padding=1, bias=True)
        #self.actnorm1 = ActNorm(self.channels_hidden, device)
        self.actnorm1 = ActNorm(self.channels_hidden)
                                
        self.conv2 = nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=(1,1), stride=1, padding=0, bias=True)
        #self.actnorm2 = ActNorm(self.channels_hidden, device)
        self.actnorm2 = ActNorm(self.channels_hidden)
        
        self.conv3 = nn.Conv2d(self.channels_hidden, channels_out, kernel_size=(3,3), stride=1, padding=1, bias=True)        
        self.logs = nn.Parameter(torch.zeros(channels_out, 1, 1))
        
        # initializing
        with torch.no_grad():
            
            nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv1.bias)
            
            nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv2.bias) 
            
            if init_last_zeros:
                nn.init.zeros_(self.conv3.weight)
            else:
                nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv3.bias)
            
        # to device
        self.to(device)
            
    def forward(self, x):
        x = self.conv1(x)
        #x, _, _ = self.actnorm1(x, logdet=0, reverse=False) 
        x = self.actnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x, _, _ = self.actnorm2(x, logdet=0, reverse=False)
        x = self.actnorm2(x) 
        x = F.relu(x)
        x = self.conv3(x)
        x = x * torch.exp(self.logs * 3)
        return x
    
        
if __name__ == "__main__":
    size = (15,9,128,128)
    net = FunctionNet(channels_in=9, channels_out=3, device=device, init_last_zeros=True)
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    for i in range(2):
        opt.zero_grad()
        x = torch.tensor(np.random.normal(0, 1, size),dtype=torch.float, device=device)
        y_true = net(x)*2 + 2
        y = net(x)
        loss = torch.norm(y - y_true)
        mu = y.mean().item()
        std = y.std().item()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print("\r loss=%0.3f| mu=%0.3f | std=%0.3f"%(loss.item(),mu,std), end="\r")

    print(x.shape, y.shape)
            
    