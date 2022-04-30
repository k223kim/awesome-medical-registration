import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans_ct
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import yaml
import csv

def main(myconfig):
    test_dir = myconfig["dataset_dir"]["value"]
    test_dir_data = os.path.join(test_dir, myconfig["input_type"]["value"])
    test_dir_label = os.path.join(test_dir, myconfig["label_type"]["value"])
    seg_test_data = os.path.join(myconfig["seg_path"]["value"], myconfig["input_type"]["value"])
    seg_test_label = os.path.join(myconfig["seg_path"]["value"], myconfig["label_type"]["value"])
    model_idx = -1
    weights = [1, 0.02]
    model_dir = myconfig["model_path"]["value"]
    out_dir = myconfig["save_path"]["value"]    
    # dict = utils.process_label()
    # if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
    #     os.remove('experiments/'+model_folder[:-1]+'.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    # line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])

    config = CONFIGS_TM['TransMorph']
    #update config accordingly
    config.img_size = tuple(map(int, myconfig["img_size"]["value"].split(', ')))
    config.in_chans = myconfig["in_chans"]["value"]
    config.window_size = tuple(map(int, myconfig["window_size"]["value"].split(', ')))
    config.embed_dim = myconfig["embed_dim"]["value"]

    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir)['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    # reg_model = utils.register_model((512, 512, 64), 'nearest')
    # reg_model.cuda()
    test_composed = trans_ct.Compose([trans_ct.NumpyType((np.float32, np.float32)),
                                        trans_ct.ToTensor(),
                                        ])
    test_set = datasets.NECTCECTTestDataset(glob.glob(test_dir_data + '/*.npz'), glob.glob(test_dir_label + '/*.npz'),transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_ssim_gt = utils.AverageMeter()
    eval_ssim_pred = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        # stdy_idx = 0
        header = ["patient", "gt-ssim", "pred-ssim", "jacobian"]
        with open(out_dir+"/swinmorph_score.csv", "w+") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for data in test_loader:
                model.eval()

                x = data[0]
                y = data[1]
                x_path = data[2][0]
                y_path = data[3][0]            

                x_name_ = x_path.split('/')
                x_name = x_name_[-1]
                x_name = x_name.replace('.npz', '')
                y_name_ = y_path.split('/')
                y_name = y_name_[-1]
                y_name = y_name.replace('.npz', '')    
                patient_num = "_".join(x_name.split("_")[1:])
                x = x.cuda()
                y = y.cuda()

                x_in = torch.cat((x,y),dim=1)
                x_def, flow = model(x_in)
                x_out = x_def.detach().cpu().numpy()[0, 0, :, :, :]#prediction
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]#groundtruth
                jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
                det_nz = np.sum(jac_det <= 0)/np.prod(tar.shape)
                np.savez_compressed(out_dir+"/"+x_name+"_prediction", data=x_out)   
                np.savez_compressed(out_dir+"/"+y_name+"_gt", data=tar)               

                ssim_pred = losses.ssim3D(x_def, y)
                ssim_gt = losses.ssim3D(x, y)

                record = [patient_num, ssim_gt, ssim_pred, det_nz]
                writer.writerow(record)
                print('ground truth ssim: {:.4f}, prediction ssim: {:.4f}'.format(ssim_gt.item(),ssim_pred.item()))
                eval_ssim_gt.update(ssim_gt.item(), x.size(0))
                eval_ssim_pred.update(ssim_pred.item(), x.size(0))
                eval_det.update(det_nz, x.size(0))

            #add total info
            sub_header = ["type", "avg", "std"]
            writer.writerow(sub_header)
            detailed_info1 = ["ssim_gt", eval_ssim_gt.avg, eval_ssim_gt.std]
            writer.writerow(detailed_info1)
            detailed_info2 = ["ssim_pred", eval_ssim_pred.avg, eval_ssim_pred.std]
            writer.writerow(detailed_info2)
            detailed_info3 = ["jacobian", eval_det.avg, eval_det.std]
            writer.writerow(detailed_info3)

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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
    with open("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/TransMorph/test_transmorph.yaml") as file:
        myconfig = yaml.safe_load(file)
    main(myconfig)