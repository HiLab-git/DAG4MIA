# -*- coding: utf-8 -*-
# @Time    : 2021/5/6 16:48
# @Author  : Ran.Gu
# @Email   : guran924@std.uestc.edu.cn
'''
This code is for 'Contrastive Semi-supervised Learning for Cross Anatomical Structure Domain Adaptative Segmentation'.
We used mean teacher as the backbone for semi-supervised domain adaptation across organs with similar shape structures.
While we introduced contrastive learning to encourage the source and target to be consistent in a latent space.
'''
import sys
import math
import os, random, torch, shutil, logging
from tqdm import tqdm
from settings import Settings
import numpy as np
import time
## experimental seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')
###
from utils import ramps, losses
from dataloader.ms_cmr.make_datalist import split_filelist
from dataloader.ms_fundus.Refuge_dataloader import RefugeDataset, ToTensor
from dataloader.ms_cmr.Cmr_dataloader import CmrsegDataset, TwoStreamBatchSampler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from models.networks.cscada_net import Unet_dsbn_cont
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.chdir(sys.path[0])
def _val_on_the_fly(model:nn.Module, loader_val:list, writer, iter_num):
    '''
    validation, and save the results in the writer
    :param model: segmentation model
    :param loader_val: validation loader: list
    :param writer: summarywriter
    :param iter_num: current iteration
    :return: loss
    '''
    model.eval()
    bce_criterion = nn.BCELoss()
    loss_bce_val = 0
    dice_val = 0

    # prediction by a patient
    for num, batch_val in enumerate(loader_val):
        img_val, gt_val = batch_val['image'].cuda(), batch_val['label'].cuda()
        with torch.no_grad():
            pred_val, _ = model(img_val, domain_label=1)
        pred_val = F.softmax(pred_val, dim=1)
        soft_gt = losses.get_soft_label(gt_val, net_params['num_classes'])
        loss_bce_val += bce_criterion(pred_val, soft_gt.permute(0, 3, 1, 2))
        pred_mask = torch.argmax(pred_val, dim=1, keepdim=True)
        pred_softmask = losses.get_soft_label(pred_mask, net_params['num_classes'])
        dice_val += losses.val_dice_class(pred_softmask[..., 1:], soft_gt[..., 1:], net_params['num_classes']-1)

        grid_image = make_grid(img_val, nrow=train_params['batch_size'], normalize=True)
        writer.add_image('val/images', grid_image, iter_num)
        grid_image = make_grid(gt_val, nrow=train_params['batch_size'], normalize=True)
        writer.add_image('val/ground_truths', grid_image, iter_num)
        grid_image = make_grid(pred_mask.float(), nrow=train_params['batch_size'], normalize=True)
        writer.add_image('val/preds', grid_image, iter_num)
    loss_bce_val /= len(loader_val)
    dice_val_class = dice_val / len(loader_val)
    loss_dice_val = 1 - torch.mean(dice_val_class)

    # summarywriter
    writer.add_scalar('val/loss_bce', loss_bce_val, iter_num)
    writer.add_scalar('val/loss_dice', loss_dice_val, iter_num)
    # for tag, value in model.named_parameters():
    #     tag = tag.replace('.', '/')
    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), iter_num)
    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), iter_num)

    return loss_bce_val, loss_dice_val, dice_val_class


def train(model:nn.Module, ema_model:nn.Module, loader_train_s:list, 
          loader_train_t:list, loader_valid_t:list, train_params:dict, writer):
    # define optimizer
    if train_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'], 
                            weight_decay=train_params['weight_decay'])
    elif train_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=train_params['learning_rate'], momentum=train_params['momentum'],
                            weight_decay=train_params['weight_decay'])

    # define losses
    dice_criterion = losses.DiceLoss()
    similar_criterion = nn.CosineSimilarity()
    if train_params['consistency_type'] == 'mse':
        consistency_criterion = losses.mse_loss
    elif train_params['consistency_type'] == 'kl':
        consistency_criterion = losses.kl_loss
    else:
        assert False, train_params['consistency_type']

    # iterator
    loader_t_iter = iter(loader_train_t)
    logging.info("{} itertations per epoch".format(len(loader_train_s)))
    
    iter_num = 0
    max_epoch = train_params['iterations'] // len(loader_train_s)
    lr_ = train_params['learning_rate']
    best_val_dice = torch.tensor(0).float()
    best_val_step = 0
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for k, batch_s in enumerate(loader_train_s):
            try:
                batch_t = next(loader_t_iter)
            except StopIteration:
                loader_t_iter = iter(loader_train_t)
                batch_t = next(loader_t_iter)

            time2 = time.time()

            # source and target data
            image_s, label_s = batch_s['image'].cuda(), batch_s['label'].cuda()
            image_t, label_t = batch_t['image'].cuda(), batch_t['label'].cuda()
            unlabeled_image_t = image_t[train_params['labeled_bs']:]

            noise = torch.clamp(torch.randn_like(unlabeled_image_t) * 0.05, -0.2, 0.2)
            ema_inputs_t = unlabeled_image_t + noise

            model.train()
            ema_model.train()
            predouts, high_r_s_sb  = model(image_s, domain_label=0)
            predoutt, high_r_t_tb  = model(image_t, domain_label=1)
            _, high_r_s_tb = model(image_s, domain_label=1)
            _, high_r_t_sb = model(image_s, domain_label=0)

            with torch.no_grad():
                ema_output, _ = ema_model(ema_inputs_t, domain_label=1)

            # segmentation loss
            loss_bce_s = F.cross_entropy(predouts, label_s.squeeze().long())
            predout_s = F.softmax(predouts, dim=1)
            pred_mask_s = torch.argmax(predout_s, dim=1, keepdim=True)
            soft_label_s = losses.get_soft_label(label_s, net_params['num_classes'])
            loss_dice_s = dice_criterion(predout_s.permute(0, 2, 3, 1)[..., 1:], soft_label_s[..., 1:], net_params['num_classes']-1)
            loss_s = 0.5 * (loss_bce_s + loss_dice_s)

            loss_bce_t = F.cross_entropy(predoutt[:train_params['labeled_bs']], label_t[:train_params['labeled_bs']].squeeze(dim=1).long())
            predout_t = F.softmax(predoutt, dim=1)
            soft_label_t = losses.get_soft_label(label_t[:train_params['labeled_bs']], net_params['num_classes'])
            loss_dice_t = dice_criterion(predout_t.permute(0, 2, 3, 1)[:train_params['labeled_bs']][..., 1:], soft_label_t[..., 1:], net_params['num_classes']-1)
            loss_t = 0.5 * (loss_bce_t + loss_dice_t)

            consistency_loss = 0
            consistency_weight = get_current_consistency_weight(iter_num//len(loader_train_s), max_epoch)
            consistency_dist = consistency_criterion(predout_t[train_params['labeled_bs']:], ema_output)    #(batch, 3, 256, 256)
            consistency_dist = torch.mean(consistency_dist)
            consistency_loss = consistency_dist * consistency_weight

            # contrastive loss
            pos_s2t_similar = similar_criterion(high_r_s_sb, high_r_t_tb) / train_params['temp_fac']
            den_s2t1_similar = similar_criterion(high_r_s_sb, high_r_s_tb) / train_params['temp_fac']
            den_s2t2_similar = similar_criterion(high_r_s_sb, high_r_t_sb) / train_params['temp_fac']
            contrast_loss_s2t = -torch.log(torch.exp(pos_s2t_similar)/(torch.exp(pos_s2t_similar)+
                                    torch.sum(torch.exp(den_s2t1_similar)+torch.exp(den_s2t2_similar))))

            pos_t2s_similar = similar_criterion(high_r_t_tb, high_r_s_sb) / train_params['temp_fac']
            den_t2s1_similar = similar_criterion(high_r_t_tb, high_r_t_sb) / train_params['temp_fac']
            den_t2s2_similar = similar_criterion(high_r_t_tb, high_r_s_tb) / train_params['temp_fac']
            contrast_loss_t2s = -torch.log(torch.exp(pos_t2s_similar)/(torch.exp(pos_t2s_similar)+
                                    torch.sum(torch.exp(den_t2s1_similar)+torch.exp(den_t2s2_similar))))
            
            contrast_loss_inter = (contrast_loss_s2t + contrast_loss_t2s)/2
            contrast_loss = 0.1*torch.mean(contrast_loss_inter)

            # total loss
            loss = 0.5*(loss_s+loss_t) + consistency_loss + contrast_loss

            # backforward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, train_params['ema_decay'], iter_num)

            # update ema model
            if epoch_num > train_params['ema_frozen_epoch']:
                update_ema_variables(model, ema_model, train_params['ema_decay'], iter_num)

            # tensorboard
            iter_num += 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_bce_s', loss_bce_s, iter_num)
            writer.add_scalar('loss/loss_dice_s', loss_dice_s, iter_num)
            writer.add_scalar('loss/loss_s', loss_s, iter_num)
            writer.add_scalar('loss/loss_bce_t', loss_bce_t, iter_num)
            writer.add_scalar('loss/loss_dice_t', loss_dice_t, iter_num)
            writer.add_scalar('loss/loss_t', loss_t, iter_num)
            writer.add_scalar('loss/contrast_loss', contrast_loss, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            # writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)

            # if iter_num % (len(loader_train_s)*10) == 0:
            #     grid_image = make_grid(image_s, nrow=train_params['batch_size'], normalize=True)
            #     writer.add_image('train/images_s', grid_image, iter_num)
            #     grid_image = make_grid(label_s, nrow=train_params['batch_size'], normalize=True)
            #     writer.add_image('train/ground_truths_s', grid_image, iter_num)
            #     grid_image = make_grid(pred_mask_s.float(), nrow=train_params['batch_size'], normalize=True)
            #     writer.add_image('train/preds_s', grid_image, iter_num)

            # validation
            if iter_num % len(loader_train_s) == 0:
                loss_bce_val, loss_dice_val, dice_val_class = _val_on_the_fly(model, loader_valid_t, writer, iter_num)
                logging.info('Validation --> loss_bce: %.4f; loss_dice: %.4f; mean_dice: %.4f (cup_dice: %.4f; disc_dice: %.4f)' %
                            (loss_bce_val.item(), loss_dice_val.item(), torch.mean(dice_val_class).item(), dice_val_class[0].item(), dice_val_class[1].item()))

                if torch.mean(dice_val_class) > torch.mean(best_val_dice):
                    best_val_dice = dice_val_class
                    best_val_step = iter_num
                    torch.save(model.state_dict(), os.path.join(common_params['exp_dir'], 'best_model.pth'))
                    logging.info('********** Best model (dice: %.4f; cup_dice: %.4f; disc_dice: %.4f) is updated at step %d.' %
                                (torch.mean(dice_val_class).item(), dice_val_class[0].item(), dice_val_class[1].item(),iter_num))
                else:
                    logging.info('********** Best model (dice: %.4f; cup_dice: %.4f; disc_dice: %.4f) was at step %d, current dice: %.4f.' %
                                (torch.mean(best_val_dice).item(), best_val_dice[0].item(), best_val_dice[1].item(), best_val_step, torch.mean(dice_val_class).item()))

            # print losses
            if iter_num % len(loader_train_s) == 0:
                logging.info('(Iteration %d, lr: %.6f) --> loss_s: %.4f; loss_t: %.4f; loss_bce_t: %.4f; loss_dice_t: %.4f; contrast_loss: %.4f; consistency_loss: %.4f; '
                            % (iter_num, lr_, loss_s.item(), loss_t.item(), loss_bce_t.item(), loss_dice_t.item(), contrast_loss.item(), consistency_loss.item()))

            # save the model
            if iter_num % 4500 == 0:
                save_mode_path = os.path.join(common_params['exp_dir'],
                                              'iter_%d_dice_%.4f.pth' % (iter_num, torch.mean(dice_val_class).item()))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            # change learning rate
            if iter_num % train_params['lr_decay_freq'] == 0:
                lr_ = train_params['learning_rate'] * 0.95 ** (iter_num // train_params['lr_decay_freq'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


def get_current_consistency_weight(epoch, max_epoches):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return train_params['consistency_rate'] * ramps.sigmoid_rampup(epoch, max_epoches)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == '__main__':
    # config file
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    # workspace
    shutil.rmtree(common_params['exp_dir'], ignore_errors=True)
    os.makedirs(common_params['exp_dir'], exist_ok=True)
    shutil.copy('./settings.ini', common_params['exp_dir'])
    logging.basicConfig(filename=os.path.join(common_params['exp_dir'], 'logs.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_dir'])

    if data_params['resplit_data']:
        split_filelist(data_params)

    # upload dataset
    dataset_train_s = RefugeDataset(data_list_dir=data_params['source_data_list_dir'], data_dir=data_params['source_data_dir'],
                                 train_type='train', image_type='image', transform=transforms.Compose([ToTensor()]))
    loader_train_s = DataLoader(dataset=dataset_train_s, batch_size=train_params['batch_size'], shuffle=True, num_workers=6,
                                drop_last=True, pin_memory=True)

    dataset_train_t = CmrsegDataset(data_list_dir=data_params['target_data_list_dir'], data_dir=data_params['target_data_dir'],
                                 train_type='train', image_type='image', transform=transforms.Compose([ToTensor()]))
    dataset_valid_t = CmrsegDataset(data_list_dir=data_params['target_data_list_dir'], data_dir=data_params['target_data_dir'],
                                 train_type='test', image_type='image', transform=transforms.Compose([ToTensor()]))
    
    labeled_idxs = list(range(round(dataset_train_t.__len__()*data_params['seen_target_percent'])))
    unlabled_idxs = list(range(round(dataset_train_t.__len__()*data_params['seen_target_percent']), dataset_train_t.__len__()))
    random.shuffle(unlabled_idxs)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabled_idxs, train_params['batch_size'], train_params['batch_size']-train_params['labeled_bs'])
    def worker_init_fn(worker_id):
        random.seed(1337+worker_id)
    loader_train_t = DataLoader(dataset_train_t, batch_sampler=batch_sampler, pin_memory=True, worker_init_fn=worker_init_fn)
    loader_valid_t = DataLoader(dataset_valid_t, batch_size=1, shuffle=False, num_workers=6,
                                pin_memory=True, drop_last=False)

    # summarywriter
    writer = SummaryWriter(log_dir=common_params['exp_dir'])

    # model
    def create_model(ema=False):
        model = Unet_dsbn_cont(net_params).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    # for var_name in model.state_dict():
    #     print(f'{var_name}, {model.state_dict()[var_name].shape}')

    train(model, ema_model, loader_train_s, loader_train_t, loader_valid_t, train_params, writer)
