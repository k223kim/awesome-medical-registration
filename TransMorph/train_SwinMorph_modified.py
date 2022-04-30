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
from torch.autograd import Variable
# from parallel import DataParallelModel, DataParallelCriterion
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = wbconfig.batch_size
    #get dataset paths
    train_dir = os.path.join(wbconfig.dataset_path, "train")
    train_dir_data = os.path.join(train_dir, wbconfig.input_type)
    train_dir_label = os.path.join(train_dir, wbconfig.label_type)

    val_dir = os.path.join(wbconfig.dataset_path, "val")
    val_dir_data = os.path.join(val_dir, wbconfig.input_type)
    val_dir_label = os.path.join(val_dir, wbconfig.label_type)
    seg_val_data = os.path.join(os.path.join(wbconfig.seg_path, "val"), wbconfig.input_type)
    seg_val_label = os.path.join(os.path.join(wbconfig.seg_path, "val"), wbconfig.label_type)

    save_dir = wbconfig.save_path

    now = datetime.now()
    now = now.strftime('%Y-%m-%d-%H:%M:%S')
    save_dir = os.path.join(save_dir, str(now))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    
    save_dir_x2y = os.path.join(save_dir, "x2y")
    if not os.path.exists(save_dir_x2y):
        os.makedirs(save_dir_x2y)
    save_dir_y2x = os.path.join(save_dir, "y2x")
    if not os.path.exists(save_dir_y2x):
        os.makedirs(save_dir_y2x)

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
    modelX = TransMorph.TransMorph(config)
    transform = TransMorph.SynSpatialTransform().cuda()
    diff_transform = TransMorph.DiffeomorphicTransform(time_step=7).cuda()
    com_transform = TransMorph.CompositionTransform().cuda()
    modelX.cuda()
    modelY = TransMorph.TransMorph(config)
    modelY.cuda()

    range_flow = 100
    grid = utils.generate_grid(config.img_size)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

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
    # if cont_training:
    #     epoch_start = 394
    #     model_dir = 'experiments/'+save_dir
    #     updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
    #     best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
    #     print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
    #     model.load_state_dict(best_model)
    # else:
    #     updated_lr = lr
    updated_lr = lr
    '''
    Initialize training
    '''
    train_composed = trans_ct.Compose([trans_ct.RandomHVFlip(),
                                         trans_ct.Random90Rotation(),
                                         trans_ct.NumpyType((np.float32, np.float32)),
                                         trans_ct.ToTensor(),
                                         ])
    val_composed = trans_ct.Compose([trans_ct.NumpyType((np.float32, np.float32)),
                                         trans_ct.ToTensor(),
                                        ])
                                        
    train_set = datasets.NECTCECTDataset(glob.glob(train_dir_data + '/*.npz'), glob.glob(train_dir_label + '/*.npz'), transforms=train_composed)
    val_set = datasets.NECTCECTInferDataset(glob.glob(val_dir_data + '/*.npz'), glob.glob(val_dir_label + '/*.npz'), \
                                            glob.glob(seg_val_data + '/*.npz'), glob.glob(seg_val_label + '/*.npz'),transforms=val_composed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizerX = optim.Adam(modelX.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizerY = optim.Adam(modelY.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    criterionR = losses.crossCorrelation3D(1, kernel=(9,9,9), device_num=0)#registration loss
    criterionL2 = losses.DisplacementRegularizer(energy_type='bending')#regularization term
    criterionC = torch.nn.L1Loss()#cycle loss
    criterionI = losses.crossCorrelation3D(1, kernel=(9,9,9), device_num=0)#identity loss

    loss_similarity = losses.NCC()

    #hyperparamters obtained from cycleMorph
    lambda_R = 1
    alpha = 0.1 #cycle loss from x2y
    beta = 1 #cycle loss from y2x

    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0

        for data in train_loader:
            iter_loss = 0
            idx += 1
            # if idx > 1:
            #     break
            modelX.train()
            modelY.train()
            adjust_learning_rate(optimizerX, epoch, max_epoch, lr) 
            adjust_learning_rate(optimizerY, epoch, max_epoch, lr)          
 
            ###Registration Loss
            #x2y     
            loss = 0 
            x = data[0].cuda()
            y = data[1].cuda()           
            x2y_in = torch.cat((x,y), dim=1)
            y_hat, x2y_flow = modelX(x2y_in)#out, flow
            loss_x2yR = criterionR(y_hat.cuda(), y.cuda())
            loss_x2yL2 = criterionL2(x2y_flow, y) * lambda_R
            loss = loss_x2yR + loss_x2yL2
            y_hat_copy = y_hat.detach().clone()
            x2y_flow_copy = x2y_flow.detach().clone()
            loss_copy_0 = loss.detach().clone()
            
            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} x2y R-loss {:.4f}'.format(idx, len(train_loader), loss.item()))
            del x2y_in
            del y_hat
            # del x2y_flow

            #y2x
            loss = 0       
            y2x_in = torch.cat((y,x), dim=1)
            x_hat, y2x_flow = modelY(y2x_in)#out, flow
            loss_y2xR = criterionR(x_hat.cuda(), x.cuda())
            loss_y2xL2 = criterionL2(y2x_flow, x) * lambda_R
            x_hat_copy = x_hat.detach().clone()
            y2x_flow_copy = y2x_flow.detach().clone()
            loss = loss_y2xR + loss_y2xL2
            loss_copy_1 = loss.detach().clone()

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} y2x R-loss {:.4f}'.format(idx, len(train_loader), loss.item()))
            del y2x_in
            del x_hat
            # del y2x_flow

            #syn
            loss = 0
            x2y_flow = x2y_flow_copy.cuda()
            y2x_flow = y2x_flow_copy.cuda()
            F_X_Y_half = diff_transform(x2y_flow, grid, range_flow)
            F_Y_X_half = diff_transform(y2x_flow, grid, range_flow)

            F_X_Y_half_inv = diff_transform(-x2y_flow, grid, range_flow)
            F_Y_X_half_inv = diff_transform(-y2x_flow, grid, range_flow)

            X_Y_half = transform(x, F_X_Y_half.permute(0, 2, 3, 4, 1) * range_flow, grid)
            Y_X_half = transform(y, F_Y_X_half.permute(0, 2, 3, 4, 1) * range_flow, grid)

            F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
            F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

            X_Y = transform(x, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
            Y_X = transform(y, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)            

            syn_loss1 = loss_similarity(X_Y_half, Y_X_half)
            print('Iter {} of {} syn_loss1 {:.4f}'.format(idx, len(train_loader), syn_loss1.item()))
            syn_loss2 = loss_similarity(y, X_Y) + loss_similarity(x, Y_X)
            print('Iter {} of {} syn_loss2 {:.4f}'.format(idx, len(train_loader), syn_loss2.item()))
            syn_loss3 = utils.magnitude_loss(F_X_Y_half*range_flow, F_Y_X_half*range_flow)
            print('Iter {} of {} syn_loss3 {:.4f}'.format(idx, len(train_loader), syn_loss3.item()))
            # syn_loss4 = utils.neg_Jdet_loss(F_X_Y.permute(0,2,3,4,1)*range_flow, grid) + utils.neg_Jdet_loss(F_Y_X.permute(0,2,3,4,1)*range_flow, grid)
            # print('Iter {} of {} syn_loss4 {:.4f}'.format(idx, len(train_loader), syn_loss4.item()))
            
            syn_loss5 = utils.smoothloss(x2y_flow*range_flow) + utils.smoothloss(y2x_flow*range_flow)
            print('Iter {} of {} syn_loss5 {:.4f}'.format(idx, len(train_loader), syn_loss5.item()))

            # loss = syn_loss1 + syn_loss2 + (0.001) * syn_loss3 + (1000 * syn_loss4) + (3 * syn_loss5)
            loss = syn_loss1 + syn_loss2 + (0.001) * syn_loss3 + syn_loss5
            loss = (-1)*Variable(loss, requires_grad = True)
            print('Iter {} of {} syn_loss_tot {:.4f}'.format(idx, len(train_loader), loss.item()))
            syn_loss_tot_copy = loss.detach().clone()       

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()                 

            del x2y_flow
            del y2x_flow
            del F_X_Y_half
            del F_Y_X_half
            del F_X_Y_half_inv
            del F_Y_X_half_inv
            del X_Y_half
            del Y_X_half
            del F_X_Y
            del F_Y_X
            del X_Y
            del Y_X
            del syn_loss1
            del syn_loss2
            del syn_loss3
            # del syn_loss4
            del syn_loss5
            del loss



            ###cycle loss
            #x_hat2y_hat
            loss = 0
            x_hat = x_hat_copy.cuda()
            y_hat = y_hat_copy.cuda()
            xh2yh_in = torch.cat((x_hat, y_hat), dim=1)
            y_tilda, xh2yh_flow = modelX(xh2yh_in)
            loss_xh2yhC = criterionC(y_tilda.cuda(), y.cuda()) * alpha
            loss = loss_xh2yhC
            loss_copy_2 = loss.detach().clone()

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} xh2yh C-loss {:.4f}'.format(idx, len(train_loader), loss.item()))
            del xh2yh_in
            del y_tilda
            del xh2yh_flow

            #y_hat2x_hat
            loss = 0
            x_hat = x_hat_copy.cuda()
            y_hat = y_hat_copy.cuda()           
            yh2xh_in = torch.cat((y_hat, x_hat), dim=1)
            x_tilda, yh2xh_flow = modelY(yh2xh_in)
            loss_yh2xhC = criterionC(x_tilda.cuda(), x.cuda()) * alpha
            loss = loss_yh2xhC
            loss_copy_3 = loss.detach().clone()

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} yh2xh C-loss {:.4f}'.format(idx, len(train_loader), loss.item()))
            del yh2xh_in
            del x_tilda
            del yh2xh_flow

            ###Identity Loss
            #x2x
            x2x_in = torch.cat((x,x), dim=1)
            x_prime, x2x_flow = modelY(x2x_in)
            loss_idx = criterionI(x_prime.cuda(), x.cuda()) * beta
            loss = loss_idx
            loss_copy_4 = loss.detach().clone()

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} x2x I-loss {:.4f}'.format(idx, len(train_loader), loss.item()))

            del x2x_in
            del x_prime
            del x2x_flow

            #y2y
            y2y_in = torch.cat((y,y), dim=1)
            y_prime, y2y_flow = modelX(y2y_in)
            loss_idy = criterionI(y_prime.cuda(), y.cuda()) * beta
            loss = loss_idy
            loss_copy_5 = loss.detach().clone()

            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} y2y I-loss {:.4f}'.format(idx, len(train_loader), loss.item()))

            del y2y_in
            del y_prime
            del y2y_flow

            iter_loss = loss_copy_0 + loss_copy_1 + loss_copy_2 + loss_copy_3 + loss_copy_4 + loss_copy_5 + syn_loss_tot_copy

            loss_all.update(iter_loss.item(), y.numel())

            print('Iter {} of {} loss {:4f}'.format(idx, len(train_loader), iter_loss))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        wandb.log({"epoch": epoch, "training_loss": loss_all.avg})
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                modelX.eval()
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                img, flow = modelX(x_in.cuda())
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
                dsc = utils.dice_val_no_class(def_out, y_seg)
                eval_dsc.update(dsc.item(), x.size(0))
                print("dsc : {}".format(eval_dsc.avg))
        best_dsc = max(eval_dsc.avg, best_dsc)
        wandb.log({"val_best_dsc": best_dsc})
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': modelX.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizerX.state_dict(),
        }, save_dir=save_dir_x2y, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': modelY.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizerY.state_dict(),
        }, save_dir=save_dir_y2x, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))        

        plt.switch_backend('agg')
        grid_fig = nectcect_fig(def_grid)
        wandb.log({"grid_fig": grid_fig})
        plt.close(grid_fig)
        pred_fig_full = nectcect_fig(img.cpu())
        pred_fig_seg = nectcect_fig(def_out)
        wandb.log({"prediction_full": pred_fig_full})
        wandb.log({"prediction_seg": pred_fig_seg})
        plt.close(pred_fig_full)  
        plt.close(pred_fig_seg)        

        x_fig_full = nectcect_fig(x.cpu())
        x_fig_seg = nectcect_fig(x_seg.cpu())
        wandb.log({"input_image_full": x_fig_full})
        wandb.log({"input_image_seg": x_fig_seg})
        plt.close(x_fig_full)
        plt.close(x_fig_seg)

        tar_fig_full = nectcect_fig(y.cpu())
        tar_fig_seg = nectcect_fig(y_seg.cpu())
        wandb.log({"ground_truth_full": tar_fig_full})
        wandb.log({"ground_truth_seg": tar_fig_seg})
        plt.close(tar_fig_full)
        plt.close(tar_fig_seg)

        loss_all.reset()

def nectcect_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, :, 32:48]
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
    torch.save(state, save_dir+"/"+filename)
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
    wandb.init(project="swinmorph_modified", config="/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/TransMorph/train_swinmorph_modified.yaml")
    wbconfig = wandb.config    
    main(wbconfig)