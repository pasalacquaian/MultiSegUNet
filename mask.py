import os, sys, glob
import numpy as np
import dicom
import math
import matplotlib.pyplot as plt
from skimage.draw import polygon
from tqdm import tqdm
from rt_utils import RTStructBuilder
from predict import predict_mask
import os.path
from os import path
import argparse
from detectB import detect

names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']
processed_path = "/home/seenia/allen/Final/processing/dataProcessed/"
bbox_saving_path = "/home/seenia/allen/Final/processing/sampleP"
result_path = "/home/seenia/allen/Final/processing/resultsP/"

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#PRE PROCESS STEPS
def normalize(im_input):
    #im_output = im_input + 1000 # We want to have air value to be 0 since HU of air is -1000
    #Intensity crop
    #im_output[im_output < 0] = 0
    #im_output[im_output > 2000] = 2000 # Kind of arbitrary to select the range from -1000 to 600 in HU
    #im_output = im_output / 2000.0
    minv = im_input.min()
    maxv = im_input.max()
    im_input = np.float32((im_input - minv)*1.0 / (1.0*(maxv - minv))) 
    return im_input

def get_hu_values(image,slices):
    image = image.astype(np.int32)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    image = np.array(image, dtype=np.int32)
    image[image < -1024] = -1024
    image[image>600] = 600
    image = normalize(image)
    return image

def read_images_masks(patient):
    image = []
    for subdir, dirs, files in os.walk(patient):
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))
        if len(dcms) == 1:
            structure = dicom.read_file(os.path.join(subdir, files[0]))
            contours = read_structure(structure)
        elif len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            image = np.stack([s.pixel_array for s in slices], axis=-1)
    #label = get_mask(contours, slices,image)
    image = get_hu_values(image,slices)
    image = image.transpose(2,0,1)
    return image

def full_process():
    SRC_DATA = opt.source
    print('Preprocessing starts........ ', end="")
    patients = [os.path.join(SRC_DATA,name) for name in os.listdir(SRC_DATA) if os.path.isdir(os.path.join(SRC_DATA, name))]
    patient = patients[1]
    name_idx = len(SRC_DATA)
    image = read_images_masks(patient)

    #Predicting and cutting bounding box
    for slice in range(image.shape[0]):
        plt.imsave(bbox_saving_path+"/hi"+str(slice)+".jpg", image[slice], cmap="gray")
    with HiddenPrints():
        result = detect("/home/seenia/allen/Final/processing/sampleP","/home/seenia/allen/Final/demo/best.pt")
    minxP, minyP, maxxP, maxyP = 1000, 1000, 0, 0
    for i in result:
        if(len(i) == 4):
            minxP, minyP, maxxP, maxyP = min(minxP,int(i[0])),min(minyP,int(i[1])),max(maxxP,int(i[2])),max(maxyP,int(i[3]))

    #Saving npy
    image = image[:,minyP-5:maxyP+5,minxP-5:maxxP+5]
    saving_name = processed_path+patient[name_idx:]+"image.npy"
    np.save(saving_name,image)
    print('Preprocessing Finished')

    #Prediction
    print('Prediction starts........ ', end="")
    predict_mask(saving_name,opt.weights)
    print('Prediction Finished')

    #Saving results
    print('Saving results starts........ ', end="")
    rtstruct = RTStructBuilder.create_new(dicom_series_path=opt.source)
    names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']
    colour = [[197,165,145],[197,165,145],[127,150,88],[253,135,192],[85,188,255]]
    with HiddenPrints():
        for organ,clr in zip(names,colour):
            result = np.load(result_path+organ+".npy")
            result = result > 0
            result = result.transpose(1,2,0)
            #if(arr.count(organ[0].lower())>0):
            rtstruct.add_roi(
              mask = result, 
              color = clr, 
              name = organ
            )
    print('Saving results Finished')
    rtstruct.save(opt.dest+'final')
    #removing all unwanted files
    stream1 = os.popen('rm -r /home/seenia/allen/Final/processing/sampleP/*.jpg')
    stream2 = os.popen('rm -r /home/seenia/allen/Final/processing/dataProcessed/*.npy')
    stream3 = os.popen('rm -r /home/seenia/allen/Final/processing/resultsP/*.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/seenia/allen/Final/demo/Unet_3D_DiceLoss_epochs_40_trainsize_45_loss_dice_trainloss_4.214956012864312', help='path to model weights')
    parser.add_argument('--source', type=str, default='/home/seenia/allen/Final/demo/data/', help='path to dicom series')
    parser.add_argument('--dest', type=str, default='/home/seenia/allen/Final/', help='path to destination')
    
    opt = parser.parse_args()
    flag = 0
    
    if(path.exists(opt.weights) & path.exists(opt.source) & path.exists(opt.dest)):
        try:
            demo = RTStructBuilder.create_new(dicom_series_path=opt.source)
            flag = 1
        except:
            print("No DICOM series found in input path")
    else:
       print ("Weights File exists:" + str(path.exists(opt.weights)))
       print ("Source File exists:" + str(path.exists(opt.source)))
       print ("Destination File exists:" + str(path.exists(opt.dest)))
    
    if(flag == 1):
        full_process()
