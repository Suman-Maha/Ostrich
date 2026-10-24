from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from pytorch_msssim import ssim, SSIM, ms_ssim, MS_SSIM


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)



def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)



def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(SSIM_Loss, self).forward(img1, img2))
    





