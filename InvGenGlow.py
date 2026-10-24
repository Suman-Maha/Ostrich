import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from math import log, pi
from scipy import linalg as la
from actnorm import ActNorm
from coupling import AdditiveCoupling

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
logabs = lambda x: torch.log(torch.abs(x))


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)

        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_s = w_s.copy()

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)
        
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)

        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out
    

# class AffineCoupling(nn.Module):
#     def __init__(self, in_channel, filter_size=512, affine=True, scale_value=4):
#         super().__init__()

#         self.affine = affine

#         self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale_value)

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channel * 8, filter_size, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filter_size, filter_size * 2, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filter_size * 2, filter_size * 2, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filter_size * 2, filter_size, 1),
#             nn.ReLU(inplace=True),
#             ZeroConv2d(filter_size, in_channel * 16 if self.affine else in_channel * 8),
#         )

#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_value)

#         self.net[0].weight.data.normal_(0, 0.05)
#         self.net[0].bias.data.zero_()

#         self.net[2].weight.data.normal_(0, 0.05)
#         self.net[2].bias.data.zero_()

#     def forward(self, input):
#         unshuffled = self.pixel_unshuffle(input)
#         #in_a, in_b = input.chunk(2, 1)
#         in_a, in_b = unshuffled.chunk(2, 1)

#         if self.affine:
#             log_s, t = self.net(in_a).chunk(2, 1)
#             # s = torch.exp(log_s)
#             #s = F.sigmoid(log_s + 2)
#             s = torch.sigmoid(log_s + 2)
#             # out_a = s * in_a + t
#             out_b = (in_b + t) * s

#         else:
#             net_out = self.net(in_a)
#             out_b = in_b + net_out

#         output = torch.cat([in_a, out_b], 1)

#         shuffled = self.pixel_shuffle(output)

#         return shuffled

#     def reverse(self, output):
#         unshuffled = self.pixel_unshuffle(output)
#         out_a, out_b = unshuffled.chunk(2, 1)

#         if self.affine:
#             log_s, t = self.net(out_a).chunk(2, 1)
#             # s = torch.exp(log_s)
#             #s = F.sigmoid(log_s + 2)
#             s = torch.sigmoid(log_s + 2)
#             # in_a = (out_a - t) / s
#             in_b = out_b / s - t

#         else:
#             net_out = self.net(out_a)
#             in_b = out_b - net_out

#         input_tensor = torch.cat([out_a, in_b], 1)
#         shuffled = self.pixel_shuffle(input_tensor)

#         return shuffled


class StableAdditiveCoupling(nn.Module):
    def __init__(self, in_channel, device, scale_value=2):
        super().__init__()
        #self.affine = affine
        squeeze_dim = (in_channel * (scale_value ** 2))

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale_value)

        self.coupling  = AdditiveCoupling(channels=squeeze_dim, device=device, nn_init_last_zeros=False)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_value)

        #self.net[0].weight.data.normal_(0, 0.05)
        #self.net[0].bias.data.zero_()

        #self.net[2].weight.data.normal_(0, 0.05)
        #self.net[2].bias.data.zero_()

    def forward(self, input):
        unshuffled = self.pixel_unshuffle(input)

        coupled_output = self.coupling(unshuffled)

        shuffled_output = self.pixel_shuffle(coupled_output)

        return shuffled_output

    def reverse(self, output):
        unshuffled = self.pixel_unshuffle(output)

        coupled_input = self.coupling.reverse(unshuffled)

        shuffled_input = self.pixel_shuffle(coupled_input)

        return shuffled_input


class FlowModule(nn.Module):
    def __init__(self, in_channel, device, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.additivecoupling = StableAdditiveCoupling(in_channel, device=device)

    def forward(self, input):
        out = self.actnorm(input)
        out = self.invconv(out)
        out = self.additivecoupling(out)

        return out

    def reverse(self, output):
        input = self.actnorm.reverse(output)
        input = self.invconv.reverse(input)
        input = self.additivecoupling.reverse(input)

        return input





