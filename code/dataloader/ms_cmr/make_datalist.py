'''
This is for MS-CMRSeg challenge dataset
'''
import os, sys
import re
import math
import random
import numpy as np

os.chdir(sys.path[0])
DATA_PATH = '../../Data/MSCMR_C0_45/image'
SAVE_PATH = '.'


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def group_by_element(lst):
    result = [[]]
    length = len(lst)
    for i in range(length):
        if i < length - 1:
            if lst[i].split('_')[0] == lst[i + 1].split('_')[0]:
                result[-1].append(lst[i])
            else:
                result[-1].append(lst[i])
                result.append([])

    result[-1].append(lst[i])
    return result


def make_filelist():
    file_folder = os.path.join(DATA_PATH)
    case_folder = sorted([x for x in os.listdir(file_folder)])
    case_folder.sort(key=str2int)
    # case_folder_list = list(set(case_folder))
    # case_folder_list = sorted(case_folder_list)

    if not os.path.isdir(os.path.join(SAVE_PATH)):
        os.makedirs(os.path.join(SAVE_PATH))

    # train_list = [os.path.join(DATA_PATH.replace('..', '/mnt/lustre/guran'), site, x) for x in case_folder]
    data_list = [os.path.join(x.split('.')[0]) for x in case_folder]
    
    ## split the datalist to train, valid, test
    # datalist_group = group_by_element(data_list)

    text_save(os.path.join(SAVE_PATH, 'data_list'), data_list)


def split_filelist(data_params):
    with open(os.path.join(data_params['target_data_list_dir'], 'data_list'), 'r') as f:
        target_img_list = f.readlines()
        target_img_list = [item.replace('\n', '') for item in target_img_list]
        
        patient_list = list(set([x.split('_')[0] for x in target_img_list]))
        random.shuffle(patient_list)
        train_target_patient_list = patient_list[:math.ceil(len(patient_list)*0.1)]     # train:test=1:1
        train_target_list = [x.split('.')[0] for x in target_img_list if x.split('_')[0] in train_target_patient_list]
        text_save(os.path.join(data_params['target_data_list_dir'], 'train_ft01_list'), train_target_list)
        # valid_target_patient_list = patient_list[len(train_target_patient_list):(len(patient_list)-len(train_target_patient_list))//3+len(train_target_patient_list)]
        # valid_target_list = [x.split('.')[0] for x in target_img_list if x.split('_')[0] in valid_target_patient_list]
        # text_save(os.path.join(data_params['target_data_list_dir'], 'valid_list'), valid_target_list)
        # test_target_patient_list = patient_list[(len(patient_list)-len(train_target_patient_list))//3+len(train_target_patient_list):]
        test_target_patient_list = patient_list[len(train_target_patient_list):]        # train:test=1:1
        test_target_list = [x.split('.')[0] for x in target_img_list if x.split('_')[0] in test_target_patient_list]
        text_save(os.path.join(data_params['target_data_list_dir'], 'test_ft01_list'), test_target_list)
    print("Split date list successfully")


def text_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


def text_lb_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        if i % 2 == 1:
            s = s.replace("'", '').replace('_segmentation.nii.gz,', '_segmentation.nii.gz') + '\n'
        else:
            s = s.replace("'", '') + ','
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


if __name__ == '__main__':
    # make_filelist()
    split_filelist()
