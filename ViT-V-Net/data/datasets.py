import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np
    
class NECTCECTDataset(Dataset):
    def __init__(self, data_path, label_path, transforms):
        self.data_paths = sorted(data_path)
        self.label_paths = sorted(label_path)
        self.transforms = transforms 

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
#         path = self.paths[index]
#         x, y = pkload(path)
        x_path = self.data_paths[index]
        y_path = self.label_paths[index]
        x = np.load(x_path)['data']
        y = np.load(y_path)['data']
#         print(x_path, np.count_nonzero(x))
#         print(y_path, np.count_nonzero(y))
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.data_paths)
    
class NECTCECTInferDataset(Dataset):
    def __init__(self, data_path, label_path, seg_data_path, seg_label_path, transforms):
        self.data_paths = sorted(data_path)
        self.label_paths = sorted(label_path)
        self.seg_data_paths = sorted(seg_data_path)
        self.seg_label_paths = sorted(seg_label_path)
        self.transforms = transforms 

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
#         path = self.paths[index]
#         x, y, x_seg, y_seg = pkload(path)

        x_path = self.data_paths[index]
        y_path = self.label_paths[index]
        x = np.load(x_path)['data']
        y = np.load(y_path)['data']
        
        seg_x_path = self.seg_data_paths[index]
        seg_y_path = self.seg_label_paths[index]
        x_seg = np.load(seg_x_path)['data']
#         print(seg_x_path, np.count_nonzero(x_seg))
        y_seg = np.load(seg_y_path)['data']
#         print(seg_y_path, np.count_nonzero(y_seg))
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
#         print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
#         print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        x, x_seg, y, y_seg = self.transforms([x, x_seg, y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.data_paths)    

class NECTCECTTestDataset(Dataset):
    def __init__(self, data_path, label_path, transforms):
        self.data_paths = sorted(data_path)
        self.label_paths = sorted(label_path)
        self.transforms = transforms 

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
#         path = self.paths[index]
#         x, y = pkload(path)
        x_path = self.data_paths[index]
        y_path = self.label_paths[index]
        x = np.load(x_path)['data']
        y = np.load(y_path)['data']
#         print(x_path, np.count_nonzero(x))
#         print(y_path, np.count_nonzero(y))
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
#         return {'x': x, 'y': y, 'x_path': x_path, 'y_path':y_path}
        return x, y, x_path, y_path
    def __len__(self):
        return len(self.data_paths)     

class NECTCECTTestDataset2(Dataset):
    def __init__(self, data_path, label_path, seg_data_path, seg_label_path, transforms):
        self.data_paths = sorted(data_path)
        self.label_paths = sorted(label_path)
        self.seg_data_paths = sorted(seg_data_path)
        self.seg_label_paths = sorted(seg_label_path)            
        self.transforms = transforms 

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_path = self.data_paths[index]
        y_path = self.label_paths[index]
        x = np.load(x_path)['data']
        y = np.load(y_path)['data']

        seg_x_path = self.seg_data_paths[index]
        seg_y_path = self.seg_label_paths[index]
        x_seg = np.load(seg_x_path)['data']
#         print(seg_x_path, np.count_nonzero(x_seg))
        y_seg = np.load(seg_y_path)['data']

        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg, y, y_seg = self.transforms([x, x_seg, y, y_seg])
        # x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_seg, y_seg, x_path, y_path
        
    def __len__(self):
        return len(self.data_paths)                        
