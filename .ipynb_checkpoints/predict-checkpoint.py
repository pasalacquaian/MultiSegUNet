import numpy as np
import os
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from model import UnetModel

mask_names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']
#weight_path="/home/seenia/manu/better_results/3D_model/Unet_3D_DiceLoss_epochs_40_trainsize_45_loss_dice_trainloss_4.214956012864312"
save_path = "/home/seenia/allen/Final/processing/resultsP/"
no_classes=5 



t.backends.cudnn.enabled = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def save_mask(pred):
    pred = F.softmax(pred,dim=1)
    maxvar = t.zeros_like(pred)
    for i in range(6):
        maxvar[:,i,...] = t.argmax(pred,1)==i 
    maxvar = maxvar.cpu().numpy().reshape(pred.shape[1],pred.shape[2],pred.shape[3],pred.shape[4])#.transpose(1,0,2,3)
    #print(maxvar.shape,image.shape)
    for i in range(1,maxvar.shape[0]):
        try:    
            os.makedirs(save_path)
            np.save(save_path+mask_names[i-1]+".npy",maxvar[i])
        except FileExistsError:
            np.save(save_path+mask_names[i-1]+".npy",maxvar[i])
    print("Saved mask files...... ", end="")


def predict_mask(patient,weight_path):
    image = np.load(patient).astype(np.float32)
    image = image.reshape((1,)+image.shape)
    name = patient[patient.rindex("/")+1:]
    image = t.from_numpy(image[None,...])
    model = UnetModel(in_channels=1,out_classes=no_classes+1).to(device)
    model.load_state_dict(t.load(weight_path))
    model.eval()

    with t.no_grad():
        pred = model(image.to(device))
        save_mask(pred)
        
