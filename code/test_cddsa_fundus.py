#!/usr/bin/env python
import os
import cv2
import sys
from numpy.lib.type_check import iscomplex
import pytz
import tqdm
import torch
import random
import argparse
import numpy as np
import os.path as osp
import torch.nn.functional as F

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import utils
from dataloader.ms_fundus.fundus_dataloader import FundusSegmentation
from dataloader.ms_fundus import fundus_transforms as tr
# from scipy.misc import imsave
from utils.utils_fundus import joint_val_image, postprocessing, save_per_img
from utils.losses import *
from datetime import datetime
from models.networks.sdnet import MEncoder, AEncoder, Segmentor, Ada_Decoder
from medpy.metric import binary
torch.set_default_tensor_type('torch.FloatTensor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=123, help='random seed')
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./exp/ms_fundus/train_cddsa')
    parser.add_argument('--data_dir', type=str, default='/mnt/data1/guran/Data/ms_fundus')
    # data config
    parser.add_argument('--data_size', type=int, default=256)
    # GPU file
    parser.add_argument('-g', '--gpu', type=int, default=1)
    # test config
    parser.add_argument('--model-file', type=str, default='test1/20220929_095627.404592/models/Ep_0200_dice_0.8893.pth', help='Model path')
    parser.add_argument('--datasetTest', type=list, default=[1], help='test folder id contain images ROIs to test')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--save_imgs', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--z_length', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=8)
    args = parser.parse_args()

    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = os.path.join(args.exp_dir, args.model_file)
    output_path = os.path.join(args.exp_dir, 'test' + str(args.datasetTest[0]), args.model_file.split('/')[1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                    transform=composed_transforms_test, state='prediction')
    batch_size = args.batch_size
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 2. model
    m_encoder = MEncoder(z_length=args.z_length, in_channel=args.in_channel, img_size=args.data_size)
    a_encoder = AEncoder(in_channel=args.in_channel, width=256, height=256, ndf=16, num_output_channels=args.anatomy_channel, norm='batchnorm', upsample='bilinear')
    segmentor = Segmentor(num_output_channels=args.anatomy_channel, num_class=args.num_classes)
    decoder = Ada_Decoder(anatomy_out_channel=args.anatomy_channel, z_length=args.z_length, out_channel=args.in_channel)
    
    if torch.cuda.is_available():
        models = {'M_enc': m_encoder.cuda(), 'A_enc': a_encoder.cuda(), 'Seg': segmentor.cuda(), 'Dec': decoder.cuda()}
    
    print('==> Loading model file: %s' % (model_file))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(model_file)
    for keys, md in models.items():
        pretrained_dict = checkpoint[keys]
        model_dict = md.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        md.load_state_dict(model_dict)
        models[keys] = md

    val_cup_dice = []
    val_disc_dice = []
    total_hd_OC = []
    total_hd_OD = []
    total_asd_OC = []
    total_asd_OD = []
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    total_num = 0

    models['M_enc'].eval()
    models['A_enc'].eval()
    models['Seg'].eval()
    models['Dec'].eval()
    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        for batch in sample:
            data = batch['image']
            target = batch['label']
            img_name = batch['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            with torch.no_grad():
                a_out = models['A_enc'](data)
                prediction = models['Seg'](a_out)
                z_out, mu_out, logvar_out = models['M_enc'](data)
                # for reconstruction
                reco = models['Dec'](a_out, z_out)

            prediction = torch.nn.functional.interpolate(prediction, size=(target.size()[2], target.size()[3]), mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target.size()[2], target.size()[3]), mode="bilinear")

            target_numpy = target.data.cpu()
            imgs = data.data.cpu()
            hd_OC = 100
            asd_OC = 100
            hd_OD = 100
            asd_OD = 100
            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=args.dataset)
                # prediction_post = torch.sigmoid(prediction[i]).data.cpu().numpy()
                test_dice = val_dice_class(torch.from_numpy(prediction_post).permute(1,2,0).cuda(), target[i].permute(1,2,0), num_class=args.num_classes)
                val_cup_dice.append(test_dice[0].data.cpu().numpy())
                val_disc_dice.append(test_dice[1].data.cpu().numpy())
                if np.sum(prediction_post[0, ...]) < 1e-4:
                    hd_OC = 100
                    asd_OC = 100
                else:
                    hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                    asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                if np.sum(prediction_post[1, ...]) < 1e-4:
                    hd_OD = 100
                    asd_OD = 100
                else:
                    hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

                    asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
                total_hd_OC.append(hd_OC)
                total_hd_OD.append(hd_OD)
                total_asd_OC.append(asd_OC)
                total_asd_OD.append(asd_OD)
                total_num += 1
                if args.save_imgs:
                    for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
                        img, lt = utils.untransform(img, lt)
                        save_per_img(img.numpy().transpose(1, 2, 0),
                                    os.path.join(output_path,'test_results'),
                                    img_name[i],
                                    lp, lt, mask_path=None, ext="bmp")

    print('OC:', val_cup_dice)
    print('OD:', val_disc_dice)
    import csv
    with open(output_path+'/Dice_results.csv', 'a+') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(['Result in: '+args.model_file])
        for index in range(len(val_cup_dice)):
            wr.writerow([torch.from_numpy(val_cup_dice[index]), torch.from_numpy(val_disc_dice[index])])

    val_cup_dice_mean = np.mean(val_cup_dice)
    val_cup_dice_std = np.std(val_cup_dice)
    val_disc_dice_mean = np.mean(val_disc_dice)
    val_disc_dice_std = np.std(val_disc_dice)
    total_dice_mean = np.mean(val_cup_dice+val_disc_dice)
    total_dice_std = np.std(val_cup_dice+val_disc_dice)
    total_hd_OC_mean = np.mean(total_hd_OC)
    total_hd_OC_std = np.std(total_hd_OC)
    total_asd_OC_mean = np.mean(total_asd_OC)
    total_asd_OC_std = np.std(total_asd_OC)
    total_hd_OD_mean = np.mean(total_hd_OD)
    total_hd_OD_std = np.std(total_hd_OD)
    total_asd_OD_mean = np.mean(total_asd_OD)
    total_asd_OD_std = np.std(total_asd_OD)

    print('''\n==>val_cup_dice : {0}-{1}'''.format(val_cup_dice_mean, val_cup_dice_std))
    print('''\n==>val_disc_dice : {0}-{1}'''.format(val_disc_dice_mean, val_disc_dice_std))
    print('''\n==>val_average_dice : {0}-{1}'''.format(total_dice_mean, total_dice_std))
    print('''\n==>ave_hd_OC : {0}-{1}'''.format(total_hd_OC_mean, total_hd_OC_std))
    print('''\n==>ave_hd_OD : {0}-{1}'''.format(total_hd_OD_mean, total_hd_OD_std))
    print('''\n==>ave_asd_OC : {0}-{1}'''.format(total_asd_OC_mean, total_asd_OC_std))
    print('''\n==>ave_asd_OD : {0}-{1}'''.format(total_asd_OD_mean, total_asd_OD_std))
    with open(osp.join(output_path, 'test' + str(args.datasetTest[0]) + '_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [['batch-size: '] + [batch_size] + [args.model_file] + ['cup dice coefficence: '] + \
               [val_cup_dice_mean]+['-']+[val_cup_dice_std] + ['disc dice coefficence: '] + \
               [val_disc_dice_mean]+['-']+[val_disc_dice_std] + ['total dice coefficence: '] + \
               [total_dice_mean]+['-']+[total_dice_std] + ['average_hd_OC: '] + \
               [total_hd_OC_mean]+['-']+[total_hd_OC_std] + ['average_hd_OD: '] + \
               [total_hd_OD_mean]+['-']+[total_hd_OD_std] + ['ave_asd_OC: '] + \
               [total_asd_OC_mean]+['-']+[total_asd_OC_std] + ['average_asd_OD: '] + \
               [total_asd_OD_mean]+['-']+[total_asd_OD_std] + ['elapse time: '] + \
               [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
