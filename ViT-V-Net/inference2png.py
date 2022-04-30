import numpy as np
import os
from PIL import Image
import pandas as pd
MAX_SLICE = 128
DATAPATH = "/home/fr2zyroom/kaeunkim/ViT-V-Net-inference/medi_setting/augmentation/test/"
SAVEPATH = "/home/fr2zyroom/kaeunkim/SwinIR_input/medi_setting/augmentation/test/"
def save_as_png(arr, path, name):
    norm_arr = (arr - arr.min())/(arr.max() - arr.min())
    norm_arr = (norm_arr*255).astype(np.uint8)  
    Image.fromarray(norm_arr).save(path+"/"+name)

def pad_array(current, max_len=MAX_SLICE):
    val = int((max_len-current)/2)
    p_value = []
    if current < max_len:
        if (max_len - current) % 2 == 0: #even number
            p_value.append(val)
            p_value.append(val)
        else:
            p_value.append(val)
            p_value.append(val+1)
    return p_value

def main():
    pre_ke = pd.read_csv("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/ViT-V-Net/preprocessing/pre_ke.csv")
    aorta_ke = pd.read_csv("/home/fr2zyroom/kaeunkim/NECT_CECT-CAC-StyleTransfer/registration/ViT-V-Net/preprocessing/aorta_ke.csv")

    data_path = DATAPATH
    for root, _, images in os.walk(data_path):
        for image in images:
            if image.startswith('aorta'):
                info = image.split('_')
                patient_num_ = "".join(info[1:3])
                aorta_start = aorta_ke[aorta_ke["patient"] == int(patient_num_)]["aorta_start"].item()
                aorta_end = aorta_ke[aorta_ke["patient"] == int(patient_num_)]["aorta_end"].item()
                pre_start = pre_ke[pre_ke["patient"] == int(patient_num_)]["pre_start"].item()
                pre_end = pre_ke[pre_ke["patient"] == int(patient_num_)]["pre_end"].item()
        
                patient_num = '_'.join(info[1:3])
                aorta_path = os.path.join(SAVEPATH, "aorta")
                pre_path = os.path.join(SAVEPATH, "pre")
                
                pre_name = 'pre_'+patient_num+'_gt.npz'
                
                aorta_img = np.load(os.path.join(data_path, image))
                aorta_img_arr = aorta_img['data'] #512,512,128
                
                pre_img = np.load(os.path.join(data_path, pre_name))
                pre_img_arr = pre_img['data']
                
                start, end = 0, 0

                aorta_ori_slice = aorta_end - aorta_start + 1
                aorta_pad = pad_array(aorta_ori_slice, max_len=MAX_SLICE)
                aorta_img_valid = aorta_img_arr[:, :, aorta_pad[0]:-aorta_pad[1]]
                
                pre_ori_slice = pre_end - pre_start + 1
                pre_pad = pad_array(pre_ori_slice, max_len=MAX_SLICE)
                pre_img_valid = pre_img_arr[:,:,pre_pad[0]:-pre_pad[1]]

                #convert aorta and pre to png
                for i in range(aorta_img_valid.shape[-1]):
                    aorta_png_name = patient_num + '_aorta_'+str(i)+'.png'
                    save_as_png(aorta_img_valid[:,:,i], aorta_path, aorta_png_name)

                    pre_png_name = patient_num + '_pre_' + str(i)+'.png'
                    save_as_png(pre_img_valid[:,:,i], pre_path, pre_png_name)

if __name__ == "__main__":
    main()