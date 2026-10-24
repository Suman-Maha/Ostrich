# Ostrich
Optimal Transport Driven Asymmetric Image-to-Image Translation Model

### Article: 
S. Mahapatra and P. Maji, "Optimal Transport Driven Asymmetric Image-to-Image Translation for Nuclei Segmentation of Histological Images," in *Neural Computing and Applications (Under Review)*, Manuscript ID: NCAA-D-25-04497R2.

The codes are written in Python (Python3) with deep learning package PyTorch. The dependencies for the execution of 
the codes are as follows -

A. DEPENDENCIES:

PyTorch
numpy
scipy
tensorboard
tensorboardX

B. HARDWARE SETTING:

The model is trained on Ubuntu 20.04 LTS 64-bit OS with NVIDIA RTX A4000 with 6144 CUDA cores (CUDA version:
11.3) and 16GB storage.

C. DATA ARRANGEMENT:

The code directory should contain 2 sub-folders: 
(a) checkpoints: for saving and loading trained model states.
(b) results: for storing outputs corresponding to test set images.

N.B. - Here '<.>' denotes the folder name (user-defined). The folder named 'DataFolder' contains sample data 
arrangement. The sub-folders contain sample image patches.

D. INPUT FILE FORMAT:

(a) Input image patches : RGB histology images, resolution - 256x256 

### Training Model ###

python Training_Model.py --dataroot <data_path> --name <folder_name>  --gpu_ids 0 --model cyclefree_gan 
--display_id 0 --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256

### Testing Model ###

python Testing_Model.py --dataroot <data_path> --name <folder_name> --gpu_ids 0 --display_id 0 --loadSize 256 
--fineSize 256

Note: Here '<.>' denotes user-defined entries. "folder_name" should be same in both training and testing. 
Similarly, "data_path" should be same in both the cases.


