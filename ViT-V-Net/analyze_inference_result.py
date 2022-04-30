#this is a script that can calculate the Dice score and MSE for the inference results
import torch
import csv
import numpy as np
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import glob

def get_inference_list(path):
    result = glob.glob(path + '/*.npz')
    result = [r.split("/")[-1] for r in result]
    return result

def dice_val(y_pred, y_true, num_clus):
    if (y_pred.sum==0) or (y_true.sum==0):
        return 1
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))

def dice_csv(config, patients):
    inference_path = config["save_path"]["value"]
    with open(os.path.join(config["inference_analyze_save_path"]["value"], "dice.csv"), 'w') as f:
        writer = csv.writer(f)
        header = ['patient', 'pred-label-dice', 'ori-label-dice']
        writer.writerow(header)
        for patient in patients:
            print("patient", patient)
            pred = config["input_type"]["value"]+"_" +patient+"_prediction.npz"
            label = config["label_type"]["value"]+"_" + patient+"_gt.npz"
            ori = config["input_type"]["value"]+"_" +patient+".npz"
            img_pred = np.load(os.path.join(inference_path, pred))
            img_label = np.load(os.path.join(inference_path, label))
            original_input_path = os.path.join(config["dataset_dir"]["value"], config["input_type"]["value"])
            ori_pre = np.load(os.path.join(original_input_path, ori))
            num_imgs = img_pred['data'].shape[-1]
            pred_label_dice, ori_label_dice = [], []
            for i in range(num_imgs):
                instance_pred = img_pred['data'][:,:,i]
                instance_label = img_label['data'][:,:,i]
                instance_ori = ori_pre['data'][:,:,i]
                if np.sum(instance_pred) != 0 and np.sum(instance_label) != 0:
                    norm_pred = (instance_pred - instance_pred.min())/(instance_pred.max() - instance_pred.min())
    #                 print("instance_pred", norm_pred.shape, norm_pred.min(), norm_pred.max())
                    norm_label = (instance_label - instance_label.min())/(instance_label.max() - instance_label.min())
    #                 print("instance_label", norm_label.shape, norm_label.min(), norm_label.max())            
                    pred_label_intersection = np.sum(norm_pred * norm_label)
    #                 print("intersection", pred_label_intersection)
                    pred_label_union = np.sum(norm_pred) + np.sum(norm_label)
    #                 print("union", pred_label_union)
                    pred_label_dice_instance = ((2.*pred_label_intersection) / (pred_label_union + 1e-5)).mean()
                    print("pred_label_dice", pred_label_dice_instance)
                    pred_label_dice.append(pred_label_dice_instance)
                if np.sum(instance_ori) != 0 and np.sum(instance_label) != 0:
                    norm_ori = (instance_ori - instance_ori.min())/(instance_ori.max() - instance_ori.min())
                    norm_label = (instance_label - instance_label.min())/(instance_label.max() - instance_label.min())
                    ori_label_intersection = np.sum(norm_ori * norm_label)
                    ori_label_union = np.sum(norm_ori) + np.sum(norm_label)
                    ori_label_dice_instance = ((2.*ori_label_intersection) / (ori_label_union + 1e-5)).mean()
                    print("ori_label_dice", ori_label_dice_instance)
                    ori_label_dice.append(ori_label_dice_instance)
            np_pred_label_dice = np.array(pred_label_dice)
            np_ori_label_dice = np.array(ori_label_dice)
            data = [patient, np_pred_label_dice.mean(), np_ori_label_dice.mean()]
            writer.writerow(data)

def plot_dice(config):
    df = pd.read_csv(os.path.join(config["inference_analyze_save_path"]["value"], "dice.csv"))
    patient_num = df['patient'].values
    x = np.arange(len(patient_num))
    w = 0.3
    plt.bar(x-w, df['pred-label-dice'].values, width=w, label='pred-label-dice')
    plt.bar(x, df['ori-label-dice'].values, width=w, label='ori-label-dice')
    plt.xticks(x, patient_num, rotation='vertical')
    # plt.xticks(x[::5],  rotation='vertical')
    plt.ylabel('dice score')
    plt.ylim([0,max(max(df['pred-label-dice']),max(df['ori-label-dice']))+0.01])
    plt.tight_layout()
    plt.xlabel('Patient Num')
    plt.legend(loc='center left', bbox_to_anchor=(1, -0.5))# fancybox=True, ncol=5)
    plt.savefig(os.path.join(config["inference_analyze_save_path"]["value"],"dice_comparison.png"), bbox_inches="tight")
    plt.close()
    # plt.show()

def mse_csv(config, patients):
    inference_path = config["save_path"]["value"]
    with open(os.path.join(config["inference_analyze_save_path"]["value"], "mse.csv"), 'w') as f:
        writer = csv.writer(f)
        header = ['patient', 'pred-label-mse', 'ori-label-mse']
        writer.writerow(header)
        for patient in patients:
            print("patient", patient)
            pred = config["input_type"]["value"]+"_" +patient+"_prediction.npz"
            label = config["label_type"]["value"]+"_" + patient+"_gt.npz"
            ori = config["input_type"]["value"]+"_" +patient+".npz"
            img_pred = np.load(os.path.join(inference_path, pred))
            img_label = np.load(os.path.join(inference_path, label))
            original_input_path = os.path.join(config["dataset_dir"]["value"], config["input_type"]["value"])
            ori_pre = np.load(os.path.join(original_input_path, ori))
            num_imgs = img_pred['data'].shape[-1]
            pred_label_mse, ori_label_mse = [], []
            for i in range(num_imgs):
                instance_pred = img_pred['data'][:,:,i]
                instance_label = img_label['data'][:,:,i]
                instance_ori = ori_pre['data'][:,:,i]
                if np.sum(instance_pred) != 0:
                    pred_label_mse_instance = np.square(np.subtract(instance_pred,instance_label)).mean()
                    print("pred_label_mse", pred_label_mse_instance)
                    pred_label_mse.append(pred_label_mse_instance)
                if np.sum(instance_ori) != 0:
                    ori_mse_instance = np.square(np.subtract(instance_ori,instance_label)).mean()
                    print("ori_mse", ori_mse_instance)
                    ori_label_mse.append(ori_mse_instance)
            np_pred_label_mse = np.array(pred_label_mse)
            np_ori_label_mse = np.array(ori_label_mse)
            data = [patient, np_pred_label_mse.mean(), np_ori_label_mse.mean()]
            writer.writerow(data)    

def plot_mse(config):
    df = pd.read_csv(os.path.join(config["inference_analyze_save_path"]["value"], "mse.csv"))
    patient_num = df['patient'].values
    x = np.arange(len(patient_num))
    w = 0.3
    plt.bar(x-w, df['pred-label-mse'].values, width=w, label='pred-label-mse')
    plt.bar(x, df['ori-label-mse'].values, width=w, label='ori-label-mse')
    plt.xticks(x, patient_num, rotation='vertical')
    # plt.xticks(x[::5],  rotation='vertical')
    plt.ylabel('MSE')
    plt.ylim([0,max(max(df['pred-label-mse']),max(df['ori-label-mse']))+0.01])
    plt.tight_layout()
    plt.xlabel('Patient Num')
    plt.legend(loc='center left', bbox_to_anchor=(1, -0.5))# fancybox=True, ncol=5)
    plt.savefig(os.path.join(config["inference_analyze_save_path"]["value"],"mse_comparison.png"), bbox_inches="tight")
    plt.close()
    # plt.show()


def main(config):
    inference_path = config["save_path"]["value"]
    result = get_inference_list(inference_path)
    patients = list(set('_'.join(r.split('_')[1:3]) for r in result))
    dice_csv(config, patients)
    mse_csv(config, patients)
    plot_dice(config)
    plot_mse(config)

if __name__ == "__main__":
    with open("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/ViT-V-Net/test-setting.yaml") as file:
        config = yaml.safe_load(file)
    main(config)