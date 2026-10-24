import torch
from torch.autograd import Variable, grad


def Gradient_Penalty(D, real, fake, batch_size, n_channel, weightage_gp, device):
    alpha = torch.rand(batch_size, n_channel, 1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.float().to(device)
    
    xhat = alpha * real + (1-alpha) * fake
    xhat = xhat.float().to(device)
    xhat = Variable(xhat, requires_grad = True)
    xhat_D = D(xhat)
    
    gradient = grad(outputs=xhat_D, inputs=xhat, grad_outputs=torch.ones(xhat_D.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * weightage_gp
    
    return penalty




