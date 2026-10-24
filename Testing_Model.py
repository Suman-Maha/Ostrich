import os
import torch
import torch.nn as nn
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from test_options import TestOptions
from dataset_loader import CreateDataLoader

from networks import define_G
from get_modelname import create_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

if __name__=='__main__':
    options = TestOptions().parse()
    options.nThreads = 1   # test code only supports nThreads = 1
    options.serial_batches = True
    options.no_flip = True  # no flip
    options.display_id = -1  # no visdom display

    dataloader = CreateDataLoader(options)
    dataset = dataloader.load_data()
    dataset_size = dataloader.__len__()
    print('*** Total Image Count = %d ***' % dataset_size)

    print("Model Name: ", options.model)

    model = create_model(options)
    model.setup(options)

    netG_A2B = model
    
    ###### Testing ######
    for i, test_data in enumerate(dataset):
        if i >= options.how_many:
            break
        #real_A = Variable(input_A.copy_(real))
        #netG_A2B.set_input(real)
        #netG_A2B.test()
        real_A, image_path = netG_A2B.set_input(test_data)
        #netG_A2B.set_input(real)
        fake_B = netG_A2B.forward(real_A)
        #fake_B = netG_A2B(real)
    
        file_name = image_path[0].split("/")[-1]
        print(file_name)
        #variant_name = "_gen.png"
        recon_name = file_name
        #print(recon_name)
        save_image(((fake_B * 0.5) + 0.5), f'results/' + recon_name)
        print(f"Operation on two-domain image {i + 1} is completed\n")


    
    
    
    