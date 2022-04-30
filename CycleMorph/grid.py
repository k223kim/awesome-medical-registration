import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans_ct
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.cycleMorph_model import cycleMorph
from models.cycleMorph_model import CONFIGS as CONFIGS
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

    config = CONFIGS['Cycle-Morph']
    model = cycleMorph()
    model.initialize(config)
    model = model.netG_A
    best_model = torch.load(model_dir)['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model_bilin = utils.register_model(config.inputSize, 'bilinear')
    reg_model_bilin.cuda()    
    reg_model = utils.register_model(config.inputSize, 'nearest')
    reg_model.cuda()
    test_composed = trans_ct.Compose([trans_ct.NumpyType((np.float32, np.float32)),
                                        trans_ct.ToTensor(),
                                        ])
    test_set = datasets.NECTCECTTestDataset2(glob.glob(test_dir_data + '/*.npz'), glob.glob(test_dir_label + '/*.npz'),\
                                            glob.glob(seg_test_data + '/*.npz'), glob.glob(seg_test_label + '/*.npz'),transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    with torch.no_grad():
        for data in test_loader:
            model.eval()

            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_path = data[4][0]
            y_path = data[5][0]         

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
            x_def, flow = model(x_in)
            grid_img = mk_grid_img(8, 1, (512,512,64))
            def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
            grid = def_grid.detach().cpu().numpy()[0,0,:,:,:]
            np.savez_compressed(out_dir+"/grid/"+x_name+"_gt", data=grid)      

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(512, 512, 64)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img        

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
    with open("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/CycleMorph/test_cyclemorph.yaml") as file:
        myconfig = yaml.safe_load(file)    
    main(myconfig)