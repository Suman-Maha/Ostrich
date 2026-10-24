import torch
import torch.nn as nn
from torch.nn import init
import functools
import os
from torch.optim import lr_scheduler
from spectral_norm import spectral_norm
from InvGenGlow import FlowModule
from iunets import iUNet
from combine_networks import MLP, Decoder, ColorAppearanceEncoder, StainDensityEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch' or 'spectral':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    #elif norm_type == 'spectral':
        #norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, color_dim, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'combgenunet128':
        netG = CombUnet128(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'combgenunet256':
        netG = CombUnet256(input_nc, output_nc, color_dim, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'combgenresnet9blocks':
        netG = CombResnet9(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'inv_gen':
        netG = InvGenGlow(input_nc, n_flow=1, n_block=1, device=device)
    elif which_model_netG == 'iunet_net':
        netG = iUNet(in_channels=3, channels=(64,128,256,384), architecture=(2,2,2,2), dim=2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the generator that consists of Resnet blocks between a few downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the Unet generator.
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Generator Module: Defining the Generator #
class TwoOutGenerator(nn.Module):
    def __init__(self, input_dim, dim, color_dim, num_down_block, num_down, num_res, mlp_dim):
        super(TwoOutGenerator, self).__init__()
        self.encoded_color_appearance_code = ColorAppearanceEncoder(input_dim=input_dim, dim=dim, num_down_block=num_down_block,
                                             color_dim=color_dim)
        self.encoded_stain_density_code = StainDensityEncoder(input_dim=input_dim, dim=dim, num_down=num_down, num_res=num_res)
        self.decoded = Decoder(dim=self.encoded_stain_density_code.output_dim, output_dim=input_dim, num_up=num_down,
                               num_res=num_res)
        self.mlp_params = MLP(input_dim=color_dim, dim=mlp_dim, output_dim=self.get_num_adain_params(self.decoded),
                              num_blocks=3)

    def assign_adain_params(self, ada_in_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = ada_in_params[:, :m.num_features]
                std = ada_in_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if ada_in_params.size(1) > 2 * m.num_features:
                    ada_in_params = ada_in_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        num_ada_in_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_ada_in_params += 2 * m.num_features
        return num_ada_in_params

    def encoder_block(self, image_patch):
        color_appearance_code = self.encoded_color_appearance_code(image_patch)
        stain_density_code = self.encoded_stain_density_code(image_patch)
        return color_appearance_code, stain_density_code

    def decoder_block(self, color_apperance_code, stain_density_code):
        ada_in_params = self.mlp_params(color_apperance_code)
        dummy_ada_in_params = torch.ones_like(ada_in_params)
        self.assign_adain_params(dummy_ada_in_params, self.decoded)
        image_patch = self.decoded(stain_density_code)
        return image_patch

    def forward(self, image_patch):
        color_appearance_code = self.encoded_color_appearance_code(image_patch)
        stain_density_code = self.encoded_stain_density_code(image_patch)
        image_patch_recon = self.decoder_block(color_apperance_code=color_appearance_code, stain_density_code=stain_density_code)
        return image_patch_recon


# Defines Combined Generator and Unet as CombGenUnet
class CombUnet128(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, use_dropout):
        super(CombUnet128, self).__init__()
        model = []
        model += [TwoOutGenerator(input_dim=3, dim=64, color_dim=1000, num_down_block=4, num_down=2, num_res=4, mlp_dim=256)]
        model += [UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=7, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
class CombUnet256(nn.Module):
    def __init__(self, input_nc, output_nc, color_dim, ngf, norm_layer, use_dropout):
        super(CombUnet256, self).__init__()
        model = []
        model += [TwoOutGenerator(input_dim=3, dim=64, color_dim=color_dim, num_down_block=4, num_down=2, num_res=4, mlp_dim=256)]
        model += [UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
class CombResnet9(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, use_dropout):
        super(CombResnet9, self).__init__()
        model = []
        model += [TwoOutGenerator(input_dim=3, dim=64, color_dim=1000, num_down_block=4, num_down=2, num_res=4, mlp_dim=256)]
        model += [ResnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define the constituent block for invertible generator
class Block(nn.Module):
    def __init__(self, in_channel, n_flow, device, conv_lu=True):
        super().__init__()
        #self.squeeze = nn.PixelUnshuffle(downscale_factor=2)
        #squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(FlowModule(in_channel, device=device, conv_lu=conv_lu))

        #self.unsqueeze = nn.PixelShuffle(upscale_factor=2)

    def forward(self, input):
        #out = self.squeeze(input)
        output = input

        for flow in self.flows:
            output = flow(output)

        #unsqueezed = self.unsqueeze(out)
        #return unsqueezed
        return output

    def reverse(self, output):
        #input = self.squeeze(output)
        input = output

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        #unsqueezed = self.unsqueeze(input)
        #return unsqueezed
        return input
    

# Define the invertible generator
class InvGenGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, device, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, device=device, conv_lu=conv_lu))
        self.blocks.append(Block(n_channel, n_flow, device=device))

    def forward(self, input):
        out = input

        for block in self.blocks:
            out = block(out)

        return out

    def reverse(self, output):
        input = output
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(input)

            else:
                input = block.reverse(input)

        return input


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          # kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      # kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)












