import glob
from torch.utils.tensorboard import SummaryWriter
import logging
import os, losses, utils#, nrrd
import shutil
import sys
from torch.utils.data import DataLoader
from data import datasets, trans_ct
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
# from ignite.contrib.handlers import ProgressBar
# from torchsummary import summary
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
import yaml
import csv



def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def main(config):
    test_dir = config["dataset_dir"]["value"]
    test_dir_data = os.path.join(test_dir, config["input_type"]["value"])
    test_dir_label = os.path.join(test_dir, config["label_type"]["value"])
    seg_test_data = os.path.join(config["seg_path"]["value"], config["input_type"]["value"])
    seg_test_label = os.path.join(config["seg_path"]["value"], config["label_type"]["value"])    
    model_dir = config["model_path"]["value"]
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    out_dir = config["save_path"]["value"]

    model = models.ViTVNet(config_vit, img_size=(512, 512, 64))
    best_model = torch.load(model_dir)['state_dict']
#     print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((512, 512, 64), 'nearest')
    reg_model.cuda()
    test_composed = trans_ct.Compose([#trans.Seg_norm(),
                                        trans_ct.NumpyType((np.float32, np.float32)),
                                        ])
#     test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_set = datasets.NECTCECTTestDataset2(glob.glob(test_dir_data + '/*.npz'), glob.glob(test_dir_label + '/*.npz'),\
                                            glob.glob(seg_test_data + '/*.npz'), glob.glob(seg_test_label + '/*.npz'),transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dice_gt = AverageMeter()
    eval_dice_pred = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        # stdy_idx = 0
        header = ["patient", "gt-dice", "pred-dice", "jacobian"]
        with open(out_dir+"/dice_score.csv", "w+") as f:
            writer = csv.writer(f)
            writer.writerow(header)        
            for data in test_loader:
                model.eval()
    #             import pdb;pdb.set_trace()
    #             data = [t.cuda() for t in data if isinstance(t[0], str) == 0]
                x = data[0]
                y = data[1]
    #             import pdb;pdb.set_trace()
                x_seg = data[2]
                y_seg = data[3]
                x_path = data[4][0]
                y_path = data[5][0]
                #get proper output name
                x_name_ = x_path.split('/')
                x_name = x_name_[-1]
                x_name = x_name.replace('.npz', '')
                y_name_ = y_path.split('/')
                y_name = y_name_[-1]
                y_name = y_name.replace('.npz', '')            
                patient_num = "_".join(x_name.split("_")[1:])            
                x = x.cuda()
                y = y.cuda()
                x_seg = x_seg.cuda()
                y_seg = y_seg.cuda()

                x_in = torch.cat((x,y),dim=1)
                x_def, flow = model(x_in)#prediction
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                x_out = x_def.detach().cpu().numpy()[0, 0, :, :, :]#prediction
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]#groundtruth
                np.savez_compressed(out_dir+"/"+x_name+"_prediction", data=x_out)   
                np.savez_compressed(out_dir+"/"+y_name+"_gt", data=tar)   
                jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
                det_nz = np.sum(jac_det <= 0)/np.prod(tar.shape)   

                dice_pred = utils.dice_val(def_out.long(), y_seg.long(), 2)
                dice_gt = utils.dice_val(x_seg.long(), y_seg.long(), 2)

                record = [patient_num, dice_gt, dice_pred, det_nz]
                writer.writerow(record)
                print('ground truth dice: {:.4f}, prediction dice: {:.4f}'.format(dice_gt.item(),dice_pred.item()))
                eval_dice_gt.update(dice_gt.item(), x.size(0))
                eval_dice_pred.update(dice_pred.item(), x.size(0))
                eval_det.update(det_nz, x.size(0))

            #add total info
            sub_header = ["type", "avg", "std"]
            writer.writerow(sub_header)
            detailed_info1 = ["dice_gt", eval_dice_gt.avg, eval_dice_gt.std]
            writer.writerow(detailed_info1)
            detailed_info2 = ["dice_pred", eval_dice_pred.avg, eval_dice_pred.std]
            writer.writerow(detailed_info2)
            detailed_info3 = ["jacobian", eval_det.avg, eval_det.std]
            writer.writerow(detailed_info3)                
if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    #bring config
    with open("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/ViT-V-Net/test-setting.yaml") as file:
        config = yaml.safe_load(file)
    main(config)