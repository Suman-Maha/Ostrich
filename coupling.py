import torch
import torch.nn as nn
import numpy as np
from function_net import FunctionNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class AdditiveCoupling(nn.Module):
    def __init__(self, channels, device, nn_init_last_zeros=False):
        super(AdditiveCoupling, self).__init__()
        self.channels = channels
        self.net = FunctionNet(channels_in=3*self.channels//4, channels_out=self.channels//4, device=device, init_last_zeros=nn_init_last_zeros)
        self.to(device)
            
    def forward(self, x_input):
        n,c,h,w = x_input.size()
        # stable additive coupling layer
        xa, xb, xc, xd = self.split(x_input, mode="split-by-chunk")
        x_bcd = torch.cat([xb,xc,xd], dim=1)
        t = self.net(x_bcd)
        ya = (xa + t)

        y_a_x_cd = torch.cat([ya,xc,xd], dim=1)
        t = self.net(y_a_x_cd)
        yb = (xb + t)

        y_ab_x_d = torch.cat([ya,yb,xd], dim=1)
        t = self.net(y_ab_x_d)
        yc = (xc + t)

        y_abc = torch.cat([ya,yb,yc], dim=1)
        t = self.net(y_abc)
        yd = (xd + t)

        x_output = torch.cat([ya,yb,yc,yd],dim=1)
        #return x_output, logdet
        return x_output
            
    def reverse(self, x_output):
        # reversal of stable additive coupling layer
        xa, xb, xc, xd = self.split(x_output, mode="split-by-chunk")
        x_abc = torch.cat([xa,xb,xc], dim=1)
        t = self.net(x_abc)
        yd = (xd - t)

        x_ab_y_d = torch.cat([xa,xb,yd], dim=1)
        t = self.net(x_ab_y_d)
        yc = (xc - t)

        x_a_y_cd = torch.cat([xa,yc,yd], dim=1)
        t = self.net(x_a_y_cd)
        yb = (xb - t)

        x_bcd = torch.cat([yb,yc,yd], dim=1)
        t = self.net(x_bcd)
        ya = (xa - t)

        x_input = torch.cat([ya,yb,yc,yd],dim=1) 
        return x_input
    
    def split(self, x, mode):
        if mode == "split-by-chunk":
            xa = x[:,:self.channels//4,:,:] 
            xb = x[:,self.channels//4:self.channels//2,:,:]
            xc = x[:,self.channels//2:3*self.channels//4,:,:] 
            xd = x[:,3*self.channels//4:,:,:]
            return xa, xb, xc, xd
        if mode == "split-by-alternating":
            xa = x[:,0::4,:,:].contiguous()
            xb = x[:,1::4,:,:].contiguous()
            xc = x[:,2::4,:,:].contiguous()
            xd = x[:,3::4,:,:].contiguous()
            return xa, xb, xc, xd
            
                
if __name__ == "__main__":
    size = (15, 12, 128, 128)
    coupling  = AdditiveCoupling(channels=12, device=device, nn_init_last_zeros=False)
    x = torch.tensor(np.random.normal(5, 10, size),dtype=torch.float,device=device)
    #logdet = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
    y = coupling(x)
    #x_rev = coupling(y, reverse=True)
    x_rev = coupling.reverse(y)
    print(x_rev.shape)
    loss_rev = torch.norm(x_rev - x)
    print(loss_rev.item())
    print(x.shape, y.shape, x_rev.shape)
    
    




