import glob
from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
class ParsingDataset(Dataset):
    
    def __init__(self, folder, dvs_wrapper=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dvs_wrapper = dvs_wrapper
        self.files = list(folder.glob("*/*.png"))
        if dvs_wrapper:
            self.files1 = list(folder.glob("*/*.png"))
            self.files2 = list(folder.glob("*/*.png"))
        else:
            self.files1 = self.files
        
        self.dvs_obj = DVS()
        self.label_dict = dict([('ship-a', 0), ('ship-b', 1), ('ship-c', 2), ('fork-a',3), ('fork-b',4), ('fork-c',5)])
        
    def __len__(self): return len(self.files)
        
    def __getitem__(self, ix):
            
        if not self.dvs_wrapper:
            f = self.files[ix]
            im = (cv2.imread(str(f))[:,:,:])
            resized = cv2.resize(im, (64,64), interpolation = cv2.INTER_AREA)
            resized = resized[None]
            label = os.path.split(os.path.split(f)[0])[1]
            label_val = self.label_dict[label.lower()]
            
            return torch.tensor(resized).to(self.device).float(), torch.tensor([label_val]).to(self.device)
        
        index1 = ix
        index2 = (ix+1)%len(self.files)
        
        f1 = self.files1[index1]
        f2 = self.files2[index2]
        
        label = os.path.split(os.path.split(f1)[0])[1]
        label_val = self.label_dict[label.lower()]
        im1 = (cv2.imread(str(f1))[:,:,:])
        img1 = cv2.resize(im1, (64,64), interpolation = cv2.INTER_AREA)
        
        im2 = (cv2.imread(str(f2))[:,:,:])
        img2 = cv2.resize(im2, (64,64), interpolation = cv2.INTER_AREA)
        
        
        final_img = self.dvs_obj.observation(img1,img2)
        resized = cv2.resize(final_img, (64,64), interpolation = cv2.INTER_AREA)
        resized = resized[None]
        
        return torch.tensor(final_img).to(self.device).float(), torch.tensor([label_val]).to(self.device)
        
class DVS:
    def __init__(self) -> None:
        self.change_threshold=60
        self.kernel_size=(3, 3)
        self.sigma=1
    
    def create_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

    def gaussianDiff(self, previous, current):
        previous = cv2.GaussianBlur(previous, self.kernel_size, self.sigma)
        np_previous = np.asarray(previous, dtype=np.int64)
        
        current = cv2.GaussianBlur(current, self.kernel_size, self.sigma)
        np_current = np.asarray(current, dtype=np.int64)
        
        change = np_current - np_previous
        
        return change.reshape(change.shape[0],change.shape[1],1)
    
    
    def observation(self,img1, img2):
        prev = self.create_grayscale(img1)
        current = self.create_grayscale(img2)
        change = self.gaussianDiff(prev, current)
        
        ## threshold
        dc = self.threshold(change)
        
        return dc

    def threshold(self, change):
        dc = np.ones(shape=change.shape) * 128
        dc[change >= self.change_threshold] = 255
        dc[change <= -self.change_threshold] = 0
        return dc