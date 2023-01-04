import os
import torch
import random
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch.nn.functional as F
from distutils.version import LooseVersion
from dataloader.ms_cmr.Cmr_dataloader import CmrsegDataset, ToTensor
from settings import Settings
from torchvision import transforms

from models.networks.cscada_net import Unet_dsbn_cont
from utils.losses import get_soft_label, val_dice, val_dice_class
from utils.losses import Intersection_over_Union, Intersection_over_Union_class
from utils.binary import assd, precision, recall
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from time import *
### experiment seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def create_visual_anno(anno):
    assert np.max(anno) < 7 # only 7 classes are supported, add new color in label2color_dict
    label2color_dict = {
        0: [0, 0, 0],
        1: [0,0,255],  
        2: [0, 255, 0], 
        3: [0, 0, 255], 
        4: [255, 215, 0],  
        5: [160, 32, 100],  
        6: [255, 64, 64],  
        7: [139, 69, 19],  
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

def test(test_loader, model):
    dice = []
    iou = []
    Assd_cup = []
    Assd_disc = []
    infer_time = []

    model.eval()
    for step, data_batch in enumerate(test_loader):
        img = data_batch['image'].cuda()        # data size: B x C x H x W
        gt = data_batch['label'].cuda()         # label size: B x C x H x W

        begin_time = time()
        with torch.no_grad():
            output, _ = model(img, domain_label=1)
        output = F.softmax(output, dim=1)
        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)

        output_dis = torch.argmax(output, dim=1, keepdim=True)
        output_soft = get_soft_label(output_dis, net_params['num_classes'])              # data shape: B x H x W x C
        soft_gt = get_soft_label(gt, net_params['num_classes'])
        # input_arr = np.squeeze(image.cpu().numpy()).astype(np.float32)
        label_arr = soft_gt.permute(0, 3, 1, 2).cpu().numpy().astype(np.uint8)   # data size: B x H x W x C
        # label_shw = np.squeeze(target.cpu().numpy()).astype(np.uint8)
        output_arr = output_soft.permute(0, 3, 1, 2).cpu().numpy().astype(np.uint8)      # data shape: B x C x H x W

        # save the image
        # for i in range(output_dis.shape[0]):
        #     pred_img = output_dis.squeeze(dim=1)[i, ...].cpu().numpy()
        #     pred_img = create_visual_anno(pred_img)
        #     cv2.imwrite(eval_params['snapshot_path']+'/results_img/pred_{}.jpg'.format(str(step*train_params['batch_size']+i+1)), pred_img)

        all_iou = Intersection_over_Union_class(output_soft[:, :, :, 1:], soft_gt[:, :, :, 1:], net_params['num_classes']-1)  # the iou accuracy
        all_dice = val_dice_class(output_soft[:, :, :, 1:], soft_gt[:, :, :, 1:], net_params['num_classes']-1)  # the dice accuracy
        
        if 0 != np.count_nonzero(output_arr[:, 1, :, :]):
            cup_assd = assd(output_arr[:, 1, :, :], label_arr[:, 1, :, :], voxelspacing=(1, 1.485, 1.485))
        elif 0 == np.count_nonzero(output_arr[:, 1, :, :]):
            cup_assd = 100
        Assd_cup.append(cup_assd)
        
        if 0 != np.count_nonzero(output_arr[:, 2, :, :]):
            disc_assd = assd(output_arr[:, 2, :, :], label_arr[:, 2, :, :], voxelspacing=(1, 1.485, 1.485))
        elif 0 == np.count_nonzero(output_arr[:, 2, :, :]):
            disc_assd = 100
        Assd_disc.append(disc_assd)
        dice_np = all_dice.cpu().numpy()
        dice.append(dice_np)
        # all_precision = precision(output_arr[:, 1, :, :], label_arr[:, 1, :, :])
        # all_recall = recall(output_arr[:, 1, :, :], label_arr[:, 1, :, :])

        iou_np = all_iou.cpu().numpy()
        iou.append(iou_np)

    df = pd.DataFrame(data=dice)
    df.to_csv(eval_params['snapshot_path'] + '/refine_result.csv')
    all_dice_mean = np.average(np.average(dice, axis=1))
    all_dice_std = np.std(np.average(dice, axis=1))
    
    dice_mean = np.average(dice, axis=0)
    dice_std = np.std(dice, axis=0)

    iou_mean = np.average(iou, axis=0)
    iou_std = np.std(iou, axis=0)

    all_assd_mean = np.average(Assd_cup+Assd_disc)
    all_assd_std = np.std(Assd_cup+Assd_disc)
    cup_assd_mean = np.average(Assd_cup)
    cup_assd_std = np.std(Assd_cup)
    disc_assd_mean = np.average(Assd_disc)
    disc_assd_std = np.std(Assd_disc)
    all_time = np.sum(infer_time)
    print(dice_mean, dice_std)
    print('The cmr mean Accuracy: {cmr_dice_mean: .4f}; The cmr Accuracy std: {cmr_dice_std: .4f}'.format(
        cmr_dice_mean=all_dice_mean, cmr_dice_std=all_dice_std))
    print('The LV mean dice: {cup_dice_mean: .4f}; The LV dice std: {cup_dice_std: .4f}'.format(
        cup_dice_mean=dice_mean[0], cup_dice_std=dice_std[0]))
    print('The Myo mean dice: {disc_dice_mean: .4f}; The Myo dice std: {disc_dice_std: .4f}'.format(
        disc_dice_mean=dice_mean[1], disc_dice_std=dice_std[1]))
    print('The cmr mean Assd: {cmr_dice_mean: .4f}; The cmr Assd std: {cmr_dice_std: .4f}'.format(
        cmr_dice_mean=all_assd_mean, cmr_dice_std=all_assd_std))
    print('The LV mean Assd: {cmr_assd_mean: .4f}; The LV Assd std: {cmr_assd_std: .4f}'.format(
        cmr_assd_mean=cup_assd_mean, cmr_assd_std=cup_assd_std))
    print('The Myo mean Assd: {cmr_assd_mean: .4f}; The Myo Assd std: {cmr_assd_std: .4f}'.format(
        cmr_assd_mean=disc_assd_mean, cmr_assd_std=disc_assd_std))
    print('The inference time: {time: .4f}'.format(time=all_time))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    # config file
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    # loading the dataset
    print('loading the {0} dataset ...'.format('test'))
    test_dataset = CmrsegDataset(data_list_dir=data_params['target_data_list_dir'], data_dir=data_params['target_data_dir'],
                              train_type='test', image_type='image', transform=transforms.Compose([ToTensor()]))
    testloader = Data.DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], shuffle=False)
    print('Loading is done\n')

    # define model
    def create_model(ema=False):
        model = Unet_dsbn_cont(net_params).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    # Load the trained best model
    modelname = os.path.join(eval_params['snapshot_path'], 'best_model.pth')
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)

        model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model.")
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    test(testloader, model)
