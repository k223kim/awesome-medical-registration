import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
MAX_SLICE = 64
DATAPATH = "/home/fr2zyroom/kaeunkim/transmorph_input/seg_data/test/aorta"
SAVEPATH = "/home/fr2zyroom/kaeunkim/transmorph_input/seg_data_png/test/aorta"
def save_as_png(arr, path, name):
    # norm_arr = (arr - arr.min())/(arr.max() - arr.min())
    norm_arr = (arr*255).astype(np.uint8) 
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
    patientcsv = pd.read_csv("/home/fr2zyroom/kaeunkim/prepare_dataset_CECT2NECT/patient.csv")

    data_path = DATAPATH
    for root, _, images in os.walk(data_path):
        for image in tqdm(images):
            # if image.startswith('aorta') and image.endswith("_prediction.npz"):
            # if image.startswith('aorta') and "mask" in image:
            if "true" in image:
                info = image.split('_')
                patient_num_ = "_".join(info[1:3])
                # patient_num_ = "_".join(info[3:5]).split(".")[0]
                aorta_start = patientcsv[patientcsv["patient"] == patient_num_]["aorta_start"].item()
                aorta_end = patientcsv[patientcsv["patient"] == patient_num_]["aorta_end"].item()
                pre_start = patientcsv[patientcsv["patient"] == patient_num_]["pre_start"].item()
                pre_end = patientcsv[patientcsv["patient"] == patient_num_]["pre_end"].item()
        
                # patient_num = '_'.join(info[1:3])
                patient_num = patient_num_

                aorta_path = os.path.join(SAVEPATH, "aorta")
                pre_path = os.path.join(SAVEPATH, "pre")
                
                # pre_name = 'pre_'+patient_num+'_mask_gt.npz'
                
                aorta_img = np.load(os.path.join(data_path, image))
                aorta_img_arr = aorta_img['data'] #512,512,128
                
                # pre_img = np.load(os.path.join(data_path, pre_name))
                # pre_img_arr = pre_img['data']
                
                start, end = 0, 0

                aorta_ori_slice = aorta_end - aorta_start + 1
                if aorta_ori_slice < MAX_SLICE:
                    aorta_pad = pad_array(aorta_ori_slice, max_len=MAX_SLICE)
                    aorta_img_valid = aorta_img_arr[:, :, aorta_pad[0]:-aorta_pad[1]]
                else: 
                    aorta_img_valid = aorta_img_arr
                
                # pre_ori_slice = pre_end - pre_start + 1
                # if pre_ori_slice < MAX_SLICE:
                #     pre_pad = pad_array(pre_ori_slice, max_len=MAX_SLICE)
                #     pre_img_valid = pre_img_arr[:,:,pre_pad[0]:-pre_pad[1]]
                # else:
                #     pre_img_valid = pre_img_arr

                #convert aorta and pre to png
                for i in range(aorta_img_valid.shape[-1]):
                    aorta_png_name = patient_num + '_aorta_'+str(i)+'.png'
                    save_as_png(aorta_img_valid[:,:,i], aorta_path, aorta_png_name)

                    # pre_png_name = patient_num + '_pre_' + str(i)+'.png'
                    # save_as_png(pre_img_valid[:,:,i], pre_path, pre_png_name)

if __name__ == "__main__":
    main()