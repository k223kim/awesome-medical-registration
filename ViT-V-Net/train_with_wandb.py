from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans_ct
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from natsort import natsorted
import wandb
from datetime import datetime
# import albumentation as A

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def main(config):
    batch_size = config.batch_size
    train_dir = os.path.join(config.dataset_path, "train")
    train_dir_data = os.path.join(train_dir, config.input_type)
    train_dir_label = os.path.join(train_dir, config.label_type)
    val_dir = os.path.join(config.dataset_path, "val")
    val_dir_data = os.path.join(val_dir, config.input_type)
    val_dir_label = os.path.join(val_dir, config.label_type)
    seg_data = os.path.join(config.seg_path, config.input_type)
    seg_label = os.path.join(config.seg_path, config.label_type)
    save_dir = config.save_path
    now = datetime.now()
    now = now.strftime('%Y-%m-%d-%H:%M:%S')
    save_dir = os.path.join(save_dir, str(now))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)      

    # save_log_dir = '0903_log'
    lr = config.learning_rate
    img_size=(512, 512, 64)
    epoch_start = 0
    max_epoch = config.epochs
    cont_training = False
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    reg_model = utils.register_model((512, 512, 64), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()    
    model = models.ViTVNet(config_vit, img_size=(512, 512, 64))
    if cont_training:
        epoch_start = 335
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    model.cuda()
    wandb.watch(model)
    train_composed = trans_ct.Compose([trans_ct.RandomHVFlip(),
                                         trans_ct.Random90Rotation(),
                                         trans_ct.NumpyType((np.float32, np.float32)),
                                         #trans_ct.NumpyType(np.float32)
                                         ])
    val_composed = trans_ct.Compose([#trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans_ct.NumpyType((np.float32, np.float32)),
                                       #trans_ct.NumpyType(np.float32)
                                        ])

    train_set = datasets.NECTCECTDataset(glob.glob(train_dir_data + '/*.npz'), glob.glob(train_dir_label + '/*.npz'), transforms=train_composed)
    val_set = datasets.NECTCECTInferDataset(glob.glob(val_dir_data + '/*.npz'), glob.glob(val_dir_label + '/*.npz'), \
                                            glob.glob(seg_data + '/*.npz'), glob.glob(seg_label + '/*.npz'),transforms=val_composed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.crossCorrelation3D(in_ch=1)
    criterions = [criterion]
    weights = [1]
    # prepare deformation loss
    criterions += [losses.DisplacementRegularizer(energy_type='bending')]
    weights += [0.02]
    best_dsc = 0
    # writer = SummaryWriter(log_dir='0903_log')
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
#         import pdb;pdb.set_trace()
        for data in train_loader:
            idx += 1
            # if idx >2:
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
        # writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        wandb.log({"epoch": epoch, "training_loss": loss})
        '''
        Validation
        '''
#         import pdb;pdb.set_trace()
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                # x = x.squeeze(0).permute(1, 0, 2, 3)
                # y = y.squeeze(0).permute(1, 0, 2, 3)
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, img_size)
                output = model(x_in)
                output_img = output[0]
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val_no_class(def_out, y_seg)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        wandb.log({"val_best_dsc": best_dsc})
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='/dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        # wandb.save("dsc{:.3f}.pth.tar".format(eval_dsc.avg))
        # writer.add_scalar('MSE/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig_seg = nectcect_fig(def_out)
        pred_fig_full = nectcect_fig(output_img)
        wandb.log({"prediction_seg": pred_fig_seg})
        wandb.log({"prediction_full": pred_fig_full})
        plt.close(pred_fig_seg)
        plt.close(pred_fig_full)

        x_fig_seg = nectcect_fig(x_seg)
        x_fig_full = nectcect_fig(x)
        wandb.log({"input_image_seg": x_fig_seg})
        wandb.log({"input_image_full": x_fig_full})
        plt.close(x_fig_seg)
        plt.close(x_fig_full)

        tar_fig_seg = nectcect_fig(y_seg)
        tar_fig_full = nectcect_fig(y)
        wandb.log({"ground_truth_seg": tar_fig_seg})
        wandb.log({"ground_truth_full": tar_fig_full})
        plt.close(tar_fig_seg)
        plt.close(tar_fig_full)

        grid_fig = nectcect_fig(def_grid)
        wandb.log({"grid_fig": grid_fig})
        plt.close(grid_fig)

        loss_all.reset()
    # writer.close()

def nectcect_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, :, 32:48]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[-1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(512, 512, 64)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img    

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


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
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    wandb.login()
    wandb.init(project="vit-v-net-final-aorta2pre", config="/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/ViT-V-Net/medi-setting.yaml")
    config = wandb.config
    main(config)