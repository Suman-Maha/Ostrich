import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.autograd import Variable

# Defining Adaptive Instance Normalization #
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b = x.size(0)
        c = x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = fun.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# Defining Layer Normalization #
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


# Defining Residual Block using Convolutional Block #
class ResidualBlock(nn.Module):
    def __init__(self, dim, use_batch=False, use_instance=True, use_identity=True):
        super(ResidualBlock, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                            padding_mode='reflect', bias=True),]
        if use_batch:
            model.append(nn.BatchNorm2d(num_features=dim))
        else:
            model.append(nn.InstanceNorm2d(num_features=dim) if use_instance else AdaptiveInstanceNorm2d(num_features=dim))
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                            padding_mode='reflect', bias=True),]
        if use_batch:
            model.append(nn.BatchNorm2d(num_features=dim))
        else:
            model.append(nn.InstanceNorm2d(num_features=dim) if use_instance else AdaptiveInstanceNorm2d(num_features=dim))
        model += [nn.Identity() if use_identity else nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

        
    def forward(self, x):
        return self.model(x)


# Class definition for developing sequential Residual Blocks #
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, use_batch, use_instance, use_identity):
        super(ResBlocks, self).__init__()
        model = []
        for i in range(num_blocks):
            model += [ResidualBlock(dim=dim, use_batch=use_batch, use_instance=use_instance, use_identity=use_identity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Stain Density Encoder Module: encodes the stain bound information #
class StainDensityEncoder(nn.Module):
    def __init__(self, input_dim, dim, num_down, num_res, use_instancenorm=True):
        super(StainDensityEncoder, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=input_dim, out_channels=dim, kernel_size=7, stride=1, padding=3,
                            padding_mode='reflect', bias=True),
                  nn.InstanceNorm2d(num_features=dim) if use_instancenorm else nn.BatchNorm2d(num_features=dim),
                  nn.ReLU(inplace=True)]
        for i in range(num_down):
            model += [nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1,
                                padding_mode='reflect', bias=True),
                      nn.InstanceNorm2d(num_features=dim * 2) if use_instancenorm else nn.BatchNorm2d(num_features=dim * 2),
                      nn.ReLU(inplace=True)]
            dim *= 2
        model += [ResBlocks(num_blocks=num_res, dim=dim, use_batch=False, use_instance=True, use_identity=True)]
        self.model = nn.Sequential(*model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# Color Appearance Encoder Module: encodes the stain bound information #
class ColorAppearanceEncoder(nn.Module):
    def __init__(self, input_dim, dim, num_down_block, color_dim, use_batch=False, use_instance=False):
        super(ColorAppearanceEncoder, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=input_dim, out_channels=dim, kernel_size=7, stride=1, padding=3,
                            padding_mode='reflect', bias=True),]
        if use_batch:
            model.append(nn.BatchNorm2d(num_features=dim))
        if use_instance:
            model.append(nn.InstanceNorm2d(num_features=dim))
        model += [nn.ReLU(inplace=True)]
        #model += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(2):
            model += [nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1,
                                padding_mode='reflect', bias=True),]
            if use_batch:
                model.append(nn.BatchNorm2d(num_features=dim * 2))
            if use_instance:
                model.append(nn.InstanceNorm2d(num_features=dim * 2))
            model += [nn.ReLU(inplace=True)]
            #model += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            dim *= 2
        for i in range(num_down_block - 2):
            model += [nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1,
                                padding_mode='reflect', bias=True),]
            if use_batch:
                model.append(nn.BatchNorm2d(num_features=dim))
            if use_instance:
                model.append(nn.InstanceNorm2d(num_features=dim))
            model += [nn.ReLU(inplace=True)]
            #model += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        model += [nn.AdaptiveAvgPool2d(output_size=1)]
        model += [nn.Conv2d(in_channels=dim, out_channels=color_dim, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# Decoder Module: decodes encoded color appearance and stain density codes #
class Decoder(nn.Module):
    def __init__(self, dim, output_dim, num_up, num_res):
        super(Decoder, self).__init__()
        model = []
        model += [ResBlocks(num_blocks=num_res, dim=dim, use_batch=False, use_instance=False, use_identity=True)]
        for i in range(num_up):
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=5, stride=1, padding=2,
                                padding_mode='reflect', bias=True),
                      LayerNorm(num_features=dim // 2),
                      nn.ReLU(inplace=True)]
            dim //= 2
        model += [nn.Conv2d(in_channels=dim, out_channels=output_dim, kernel_size=7, stride=1, padding=3,
                            padding_mode='reflect', bias=True),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, dim, output_dim, num_blocks):
        super(MLP, self).__init__()
        model = []
        model += [nn.Linear(in_features=input_dim, out_features=dim, bias=True),
                  nn.ReLU(inplace=True)]
        for i in range(num_blocks - 2):
            model += [nn.Linear(in_features=dim, out_features=dim, bias=True),
                      nn.ReLU(inplace=True)]
        model += [nn.Linear(in_features=dim, out_features=output_dim, bias=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


# Generator Module: Defining the Generator #
class Generator(nn.Module):
    def __init__(self, input_dim, dim, color_dim, num_down_block, num_down, num_res, mlp_dim):
        super(Generator, self).__init__()
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
        self.assign_adain_params(ada_in_params, self.decoded)
        image_patch = self.decoded(stain_density_code)
        return image_patch

    def forward(self, image_patch):
        color_appearance_code = self.encoded_color_appearance_code(image_patch)
        stain_density_code = self.encoded_stain_density_code(image_patch)
        dummy_color_appearance_code = torch.ones(color_appearance_code.shape)
        print(color_appearance_code.shape)
        print(stain_density_code.shape)
        image_patch_recon = self.decoder_block(color_apperance_code=color_appearance_code, stain_density_code=stain_density_code)
        return image_patch_recon


# Discriminator Module: defining the Discriminator #
class Discriminator(nn.Module):
    def __init__(self, input_dim, dim, num_layers, num_scales, use_leaky=True):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.num_layers = num_layers
        self.use_leaky = use_leaky
        self.disc_net = nn.ModuleList()
        for _ in range(num_scales):
            self.disc_net.append(self.make_disc_net())
        self.down_sample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def make_disc_net(self):
        dimension = self.dim
        model = []
        model += [nn.Conv2d(in_channels=self.input_dim, out_channels=dimension, kernel_size=4, stride=2, padding=1,
                            padding_mode='reflect', bias=True),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True) if self.use_leaky else nn.ReLU(inplace=True)]
        for i in range(self.num_layers - 1):
            model += [nn.Conv2d(in_channels=dimension, out_channels=dimension * 2, kernel_size=4, stride=2, padding=1,
                                padding_mode='reflect', bias=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True) if self.use_leaky else nn.ReLU(inplace=True)]
            dimension *= 2
        model += [nn.Conv2d(in_channels=dimension, out_channels=1, kernel_size=1, stride=1, padding=0)]
        disc_net = nn.Sequential(*model)
        return disc_net

    def calculate_discriminator_loss(self, input_real, input_fake):
        out_input_real = self.forward(input_real)
        out_input_fake = self.forward(input_fake)
        disc_loss = 0
        for iter_count, (out_real, out_fake) in enumerate(zip(out_input_real, out_input_fake)):
            disc_loss += torch.mean((out_real - 1)**2) + torch.mean((out_fake - 0)**2)
            #all_fake = Variable(torch.zeros_like(out_fake.data).cuda(), requires_grad=False)
            #all_real = Variable(torch.ones_like(out_real.data).cuda(), requires_grad=False)
            #disc_loss += torch.mean(fun.binary_cross_entropy(fun.sigmoid(out_fake), all_fake) + fun.binary_cross_entropy(fun.sigmoid(out_real), all_real))
        return disc_loss

    def calculate_generator_loss(self, input_fake):
        out_input_fake = self.forward(input_fake)
        gen_loss = 0
        for iter_count, out_fake in enumerate(out_input_fake):
            gen_loss += torch.mean((out_fake - 1)**2)
            #all_fake = Variable(torch.ones_like(out_fake.data).cuda(), requires_grad=False)
            #gen_loss += torch.mean(fun.binary_cross_entropy(fun.sigmoid(out_fake), all_fake))
        return gen_loss
    
    def forward(self, x):
        output = []
        for layer in self.disc_net:
            output.append(layer(x))
            x = self.down_sample(x)
        return output


# Driver function to test the whole code #
def testnet():
    color_dim = 1000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 256, 256)
    x = x.to(DEVICE)
    print('Original data: ', x.size())

    gen = Generator(input_dim=3, dim=64, color_dim=color_dim, num_down_block=4, num_down=2, num_res=4, mlp_dim=256)
    gen = gen.to(DEVICE)
    x_recon = gen(x)
    print('Generated data: ', x_recon.shape)
    
    disc = Discriminator(input_dim=3, dim=64, num_layers=4, num_scales=3, use_leaky=True)
    disc = disc.to(DEVICE)
    d_loss = disc.calculate_discriminator_loss(x, x_recon)
    g_loss = disc.calculate_generator_loss(x_recon)
    print('Discriminator Loss: ', d_loss)
    print('Generator Loss: ', g_loss)
    
    
    
    



if __name__ == "__main__":
    testnet()







    
