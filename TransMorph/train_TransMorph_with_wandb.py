from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans_ct
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import wandb
from datetime import datetime
import gc

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(wbconfig):
    batch_size = wbconfig.batch_size
    #get dataset paths
    train_dir = os.path.join(wbconfig.dataset_path, "train")
    train_dir_data = os.path.join(train_dir, wbconfig.input_type)
    train_dir_label = os.path.join(train_dir, wbconfig.label_type)
    seg_train_data = os.path.join(os.path.join(wbconfig.seg_path, "train"), wbconfig.input_type)
    seg_train_label = os.path.join(os.path.join(wbconfig.seg_path, "train"), wbconfig.label_type)
    val_dir = os.path.join(wbconfig.dataset_path, "val")
    val_dir_data = os.path.join(val_dir, wbconfig.input_type)
    val_dir_label = os.path.join(val_dir, wbconfig.label_type)
    seg_val_data = os.path.join(os.path.join(wbconfig.seg_path, "val"), wbconfig.input_type)
    seg_val_label = os.path.join(os.path.join(wbconfig.seg_path, "val"), wbconfig.label_type)
    save_dir = wbconfig.save_path
    weights = [1, 0.02]
    # save_dir = os.path.join(save_dir, 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1]))
    now = datetime.now()
    now = now.strftime('%Y-%m-%d-%H:%M:%S')
    save_dir = os.path.join(save_dir, str(now))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    save_dir_loss = os.path.join(save_dir, "loss")
    if not os.path.exists(save_dir_loss):
        os.makedirs(save_dir_loss)    

    # if not os.path.exists('experiments/'+save_dir):
    #     os.makedirs('experiments/'+save_dir)
    # if not os.path.exists('logs/'+save_dir):
    #     os.makedirs('logs/'+save_dir)
    # sys.stdout = Logger('logs/'+save_dir)
    if not os.path.exists(save_dir + '/logs'):
        os.makedirs(save_dir + '/logs')
    sys.stdout = Logger(save_dir + '/logs')
    lr = wbconfig.learning_rate # learning rate
    epoch_start = 0
    max_epoch = wbconfig.epochs #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    #update config accordingly
    config.img_size = tuple(map(int, wbconfig.img_size.split(', ')))
    config.in_chans = wbconfig.in_chans
    config.window_size = tuple(map(int, wbconfig.window_size.split(', ')))
    config.embed_dim = wbconfig.embed_dim
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 394
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = trans_ct.Compose([trans_ct.RandomHVFlip(),
                                         trans_ct.Random90Rotation(),
                                         trans_ct.NumpyType((np.float32, np.float32)),
                                         trans_ct.ToTensor()#,
                                         #trans_ct.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         ])
    val_composed = trans_ct.Compose([trans_ct.NumpyType((np.float32, np.float32)),
                                         trans_ct.ToTensor()#,
                                         #trans_ct.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
                                        
    train_set = datasets.NECTCECTDataset(glob.glob(train_dir_data + '/*.npz'), glob.glob(train_dir_label + '/*.npz'), transforms=train_composed)
    val_set = datasets.NECTCECTInferDataset(glob.glob(val_dir_data + '/*.npz'), glob.glob(val_dir_label + '/*.npz'), \
                                            glob.glob(seg_val_data + '/*.npz'), glob.glob(seg_val_label + '/*.npz'),transforms=val_composed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.crossCorrelation3D(in_ch=1)
    criterions = [criterion]
    # criterions += [losses.DisplacementRegularizer(energy_type='bending')]
    criterions += [losses.Grad3d(penalty='l2')]
    #added dice loss in order to improve
    min_loss = 1000
    min_val_loss = 1000    
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            # if idx > 1:
            #     break
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)          
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]           
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            seg_flag2 = False #this is the flag to indicate any of the segmentation masks are zero (no mask)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        wandb.log({"epoch": epoch, "training_loss": loss_all.avg})
        # save the weight when it reaches minimum loss
        min_loss = min(loss_all.avg, min_loss)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_dsc': best_dsc,
        #     'min_loss' : min_loss,
        #     'optimizer': optimizer.state_dict(),
        # }, save_dir=save_dir_loss, filename='/loss{:.3f}.pth.tar'.format(loss_all.avg))
        '''
        Validation
        '''
        val_loss = utils.AverageMeter()
        val_idx = 0
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                val_idx += 1
                # if val_idx > 2:
                #     break
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                # x_seg = data[2]
                # y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                loss = 0
                loss_vals = []
                for n, loss_function in enumerate(criterions):
                    curr_loss = loss_function(output[n], y) * weights[n]
                    loss_vals.append(curr_loss)
                    loss += curr_loss
                val_loss.update(loss.item(), y.numel())

                output_img = output[0]
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])

                x_in = torch.cat((y, x), dim=1)
                loss = 0
                output = model(x_in)
                for n, loss_function in enumerate(criterions):
                    curr_loss = loss_function(output[n], x) * weights[n]
                    loss_vals[n] += curr_loss
                    loss += curr_loss
                val_loss.update(loss.item(), y.numel())                
                # dsc = utils.dice_val(def_out.cuda().long(), y_seg.cuda().long(), 2)
                # eval_dsc.update(dsc.item(), x.size(0))
            print('Epoch {} validation loss {:.4f}'.format(epoch, val_loss.avg))
                # print(eval_dsc.avg)
        min_val_loss = min(val_loss.avg, min_val_loss)
        wandb.log({"val_loss": val_loss.avg})
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'min_val_loss': min_val_loss,
            'min_loss' : min_loss,            
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir_loss, filename='/loss{:.3f}.pth.tar'.format(val_loss.avg))
        plt.switch_backend('agg')
        pred_fig_full = nectcect_fig(output_img)
        grid_fig = nectcect_fig(def_grid)
        x_fig_full = nectcect_fig(x)
        tar_fig_full = nectcect_fig(y)
        wandb.log({"grid_fig": grid_fig})
        plt.close(grid_fig)
        wandb.log({"input_image_full": x_fig_full})
        plt.close(x_fig_full)
        wandb.log({"ground_truth_full": tar_fig_full})
        plt.close(tar_fig_full)
        wandb.log({"prediction_full": pred_fig_full})
        plt.close(pred_fig_full)
        loss_all.reset()
        val_loss.reset()

def nectcect_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, :, 48:64]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[-1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(512, 512, 64)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    wandb.login()
    wandb.init(project="transmorph-final-aorta2pre-l2", config="/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/TransMorph/train_transmorph_config.yaml")
    wbconfig = wandb.config    
    main(wbconfig)