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
import random
import imgaug
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
    save_dir_x2y_loss = os.path.join(save_dir_x2y, "loss")
    if not os.path.exists(save_dir_x2y_loss):
        os.makedirs(save_dir_x2y_loss)    
    # save_dir_x2y_dice = os.path.join(save_dir_x2y, "dice")
    # if not os.path.exists(save_dir_x2y_dice):
    #     os.makedirs(save_dir_x2y_dice)    

    save_dir_y2x = os.path.join(save_dir, "y2x")
    if not os.path.exists(save_dir_y2x):
        os.makedirs(save_dir_y2x)
    save_dir_y2x_loss = os.path.join(save_dir_y2x, "loss")
    if not os.path.exists(save_dir_y2x_loss):
        os.makedirs(save_dir_y2x_loss)     
    # save_dir_y2x_dice = os.path.join(save_dir_y2x, "dice")
    # if not os.path.exists(save_dir_y2x_dice):
    #     os.makedirs(save_dir_y2x_dice)                

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
    # criterionL2 = losses.DisplacementRegularizer(energy_type='bending')#regularization term
    criterionL2 = losses.Grad3d(penalty='l2')#regularization term
    criterionC = torch.nn.L1Loss()#cycle loss
    criterionI = losses.crossCorrelation3D(1, kernel=(9,9,9), device_num=0)#identity loss

    #hyperparamters obtained from cycleMorph
    lambda_R = 1
    alpha = 0.1 #cycle loss from x2y
    beta = 1 #cycle loss from y2x

    min_val_loss = 1000
    # min_loss = 1000
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
            loss_copy_0 = loss.detach().clone()
            
            optimizerX.zero_grad()
            optimizerY.zero_grad()
            loss.backward()
            optimizerX.step()
            optimizerY.step()
            print('Iter {} of {} x2y R-loss {:.4f}'.format(idx, len(train_loader), loss.item()))
            del x2y_in
            del y_hat
            del x2y_flow

            #y2x
            loss = 0       
            y2x_in = torch.cat((y,x), dim=1)
            x_hat, y2x_flow = modelY(y2x_in)#out, flow
            loss_y2xR = criterionR(x_hat.cuda(), x.cuda())
            loss_y2xL2 = criterionL2(y2x_flow, x) * lambda_R
            x_hat_copy = x_hat.detach().clone()
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
            del y2x_flow

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

            iter_loss = loss_copy_0 + loss_copy_1 + loss_copy_2 + loss_copy_3 + loss_copy_4 + loss_copy_5

            loss_all.update(iter_loss.item(), y.numel())

            print('Iter {} of {} loss {:4f}'.format(idx, len(train_loader), iter_loss))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        wandb.log({"epoch": epoch, "training_loss": loss_all.avg})
        # save the weight when it reaches minimum loss
        # min_loss = min(loss_all.avg, min_loss)

        '''
        Validation
        '''
        val_loss_total = utils.AverageMeter()
        val_idx=  0
        with torch.no_grad():
            for data in val_loader:
                modelX.eval()
                modelY.eval()
                val_idx += 1
                # if val_idx > 2:
                #     break
                #x2y
                val_x = data[0].cuda()
                val_y = data[1].cuda()
                # x_seg = data[2]
                # y_seg = data[3]
                grid_img = mk_grid_img(8, 1, config.img_size)

                #registration loss
                #x2y
                val_loss = 0 
                val_x2y_in = torch.cat((val_x, val_y), dim=1)
                val_y_hat, val_x2y_flow = modelX(val_x2y_in)
                val_loss_x2yR = criterionR(val_y_hat.cuda(), val_y.cuda())
                val_loss_x2yL2 = criterionL2(val_x2y_flow, val_y) * lambda_R
                val_loss = val_loss_x2yR + val_loss_x2yL2
                val_y_hat_copy = val_y_hat.detach().clone()
                val_loss_copy_0 = val_loss.detach().clone()

                def_grid = reg_model_bilin([grid_img.float(), val_x2y_flow.cuda()])

                del val_x2y_in
                del val_y_hat
                del val_x2y_flow       

                 #y2x       
                val_loss = 0       
                val_y2x_in = torch.cat((val_y,val_x), dim=1)
                val_x_hat, val_y2x_flow = modelY(val_y2x_in)#out, flow
                val_loss_y2xR = criterionR(val_x_hat.cuda(), val_x.cuda())
                val_loss_y2xL2 = criterionL2(val_y2x_flow, val_x) * lambda_R
                val_x_hat_copy = x_hat.detach().clone()
                val_loss = val_loss_y2xR + val_loss_y2xL2
                val_loss_copy_1 = val_loss.detach().clone()   

                del val_y2x_in
                del val_x_hat
                del val_y2x_flow   

                ###cycle loss
                #x_hat2y_hat
                val_loss = 0
                val_x_hat = val_x_hat_copy.cuda()
                val_y_hat = val_y_hat_copy.cuda()
                val_xh2yh_in = torch.cat((val_x_hat, val_y_hat), dim=1)
                val_y_tilda, val_xh2yh_flow = modelX(val_xh2yh_in)
                val_loss_xh2yhC = criterionC(val_y_tilda.cuda(), y.cuda()) * alpha
                val_loss = val_loss_xh2yhC
                val_loss_copy_2 = val_loss.detach().clone()    

                del val_xh2yh_in
                del val_y_tilda
                del val_xh2yh_flow              

                #y_hat2x_hat
                val_loss = 0
                val_x_hat = val_x_hat_copy.cuda()
                val_y_hat = val_y_hat_copy.cuda()           
                val_yh2xh_in = torch.cat((val_y_hat, val_x_hat), dim=1)
                val_x_tilda, val_yh2xh_flow = modelY(val_yh2xh_in)
                val_loss_yh2xhC = criterionC(val_x_tilda.cuda(), val_x.cuda()) * alpha
                val_loss = val_loss_yh2xhC
                val_loss_copy_3 = val_loss.detach().clone() 

                del val_yh2xh_in
                del val_x_tilda
                del val_yh2xh_flow          

                #x2x
                val_x2x_in = torch.cat((val_x,val_x), dim=1)
                val_x_prime, val_x2x_flow = modelY(val_x2x_in)
                val_loss_idx = criterionI(val_x_prime.cuda(), val_x.cuda()) * beta
                val_loss = val_loss_idx
                val_loss_copy_4 = val_loss.detach().clone()          

                del val_x2x_in
                del val_x_prime
                del val_x2x_flow                

                #y2y
                val_y2y_in = torch.cat((val_y,val_y), dim=1)
                val_y_prime, val_y2y_flow = modelX(val_y2y_in)
                val_loss_idy = criterionI(val_y_prime.cuda(), val_y.cuda()) * beta
                val_loss = val_loss_idy
                val_loss_copy_5 = val_loss.detach().clone()          

                del val_y2y_in
                del val_y_prime
                del val_y2y_flow                                                   

                val_iter_loss = val_loss_copy_0 + val_loss_copy_1 + val_loss_copy_2 + val_loss_copy_3 + val_loss_copy_4 + val_loss_copy_5

                val_loss_total.update(val_iter_loss.item(), val_x.size(0))
                print("val loss : {}".format(val_loss_total.avg))
        min_val_loss = min(val_loss_total.avg, min_val_loss)
        wandb.log({"val_loss": val_loss_total.avg})
        wandb.log({'min_val_loss': min_val_loss})
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': modelX.state_dict(),
            'min_val_loss': min_val_loss,           
            'optimizer': optimizerX.state_dict(),
        }, save_dir=save_dir_x2y_loss, filename='loss{:.3f}.pth.tar'.format(val_loss_total.avg))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': modelY.state_dict(),
            'min_val_loss': min_val_loss,           
            'optimizer': optimizerY.state_dict(),
        }, save_dir=save_dir_y2x_loss, filename='loss{:.3f}.pth.tar'.format(val_loss_total.avg))        

        plt.switch_backend('agg')
        grid_fig = nectcect_fig(def_grid)
        wandb.log({"grid_fig": grid_fig})
        plt.close(grid_fig)
        pred_fig_full = nectcect_fig(y_hat.cpu())
        # pred_fig_seg = nectcect_fig(def_out)
        wandb.log({"prediction_full": pred_fig_full})
        # wandb.log({"prediction_seg": pred_fig_seg})
        plt.close(pred_fig_full)  
        # plt.close(pred_fig_seg)        

        x_fig_full = nectcect_fig(x.cpu())
        # x_fig_seg = nectcect_fig(x_seg.cpu())
        wandb.log({"input_image_full": x_fig_full})
        # wandb.log({"input_image_seg": x_fig_seg})
        plt.close(x_fig_full)
        # plt.close(x_fig_seg)

        tar_fig_full = nectcect_fig(y.cpu())
        # tar_fig_seg = nectcect_fig(y_seg.cpu())
        wandb.log({"ground_truth_full": tar_fig_full})
        # wandb.log({"ground_truth_seg": tar_fig_seg})
        plt.close(tar_fig_full)
        # plt.close(tar_fig_seg)

        loss_all.reset()
        val_loss_total.reset()

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

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    imgaug.random.seed(SEED)

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
    wandb.init(project="swinmorph_original_final_l2", config="/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/TransMorph/train_swinmorph_original.yaml")
    wbconfig = wandb.config  
    # set_all_seeds(0)  
    main(wbconfig)