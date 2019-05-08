import datetime
import math
import os
import os.path as osp
import shutil
import cv2
# import fcn
import numpy as np
import pytz
import scipy.misc
import scipy.io as sio
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import resize
from scipy.spatial import distance
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
from ContrastiveLoss import ContrastiveLoss
import skimage
import random
from utils.image_pool import ImagePool
from models.utils import HookBasedFeatureExtractor
from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score


import torchsrc

def saveOneImg(img,path,cate_name,sub_name,surfix,):
    filename = "%s-x-%s-x-%s.png"%(cate_name,sub_name,surfix)
    file = os.path.join(path,filename)
    scipy.misc.imsave(file, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def weighted_center(input,threshold=0.75):
    # m= torch.nn.Tanh()
    # input = m(input)

    input = torch.add(input, -input.min().expand(input.size())) / torch.add(input.max().expand(input.size()), -input.min().expand(input.size()))
    m = torch.nn.Threshold(threshold, 0)
    input = m(input)
    # if input.sum()==0:
    #     input=input
    # mask_ind = input.le(0.5)
    # input.masked_fill_(mask_ind, 0.0)
    grid = np.meshgrid(range(input.size()[0]), range(input.size()[1]), indexing='ij')
    x0 = torch.mul(input, Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / input.sum()
    y0 = torch.mul(input, Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / input.sum()
    return x0, y0


# def max_center(input,target,pts):
#     input.max()
#     return x0, y0


def get_distance(target,score,ind,Threshold=0.75):
    dist_list = []
    coord_list = []
    target_coord_list = []
    weight_coord_list = []
    for i in range(target.size()[1]):
        targetImg = target[ind,i,:,:].data.cpu().numpy()
        scoreImg = score[ind,i,:,:].data.cpu().numpy()
        targetCoord = np.unravel_index(targetImg.argmax(),targetImg.shape)
        scoreCoord = np.unravel_index(scoreImg.argmax(),scoreImg.shape)
        # grid = np.meshgrid(range(score.size()[2]), range(score.size()[3]), indexing='ij')
        # x0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        # y0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        #
        y0,x0 = weighted_center(score[ind,i,:,:],Threshold)

        weightCoord = (x0.data.cpu().numpy()[0],y0.data.cpu().numpy()[0])
        distVal = distance.euclidean(scoreCoord,targetCoord)
        dist_list.append(distVal)
        coord_list.append(scoreCoord)
        target_coord_list.append(targetCoord)
        weight_coord_list.append(weightCoord)
    return dist_list,coord_list,target_coord_list,weight_coord_list

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=3)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=3)  # b,c,1,1

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=3)  # b,c,1,1

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total

def dice_loss_norm(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total




def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def l2_normloss(input,target,size_average=True):
    criterion = torch.nn.MSELoss().cuda()  
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss

def l2_normloss_new(input,target,mask):
    loss = input - target
    loss = torch.pow(loss,2)
    loss = torch.mul(loss, mask)
    loss = loss.sum() / mask.sum()
    return loss

def l1_normloss(input,target,size_average=True):
    criterion = torch.nn.L1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l1_smooth_normloss(input,target,size_average=True):
    criterion = torch.nn.SmoothL1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l2_normloss_compete(input,target,size_average=True):
    mask = torch.sum(target, 1)
    mask = mask.expand(input.size())
    mask_ind = mask.le(0.5)
    input.masked_fill_(mask_ind, 0.0)
    mask = torch.mul(mask, 0)
    input = torch.mul(input,10)
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(input,mask)
    return loss

def l2_normloss_all(inputs,target,category_name,all_categories):
    for i in range(len(all_categories)):
        cate = all_categories[i]
        if i == 0 :
            if category_name == cate:
                loss = l2_normloss(inputs[i],target)
            else :
                loss = l2_normloss_compete(inputs[i],target)
        else:
            if category_name == cate :
                loss += l2_normloss(inputs[i],target)
            else :
                loss += l2_normloss_compete(inputs[i],target)
    return loss

def l1_loss(input, target):
    return torch.sum(torch.abs(input - target))/target.size()[0]

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)


def rmse_loss(input, target):
    return torch.sqrt(torch.sum((input - target) ** 2))


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)



def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))




def prior_loss(input,category_name,pts,target):
    mu = {}
    std = {}
    #caculated from get_spatial_prior
    # mu['KidneyLong'] = [210.420535]
    # std['KidneyLong'] = [25.846215]
    # mu['KidneyTrans'] = [104.701820, 96.639190]
    # std['KidneyTrans'] = [17.741928, 19.972482]
    # mu['LiverLong'] = [303.206934]
    # std['LiverLong'] = [45.080338]
    # mu['SpleenLong'] = [202.573985]
    # std['SpleenLong'] = [39.253982]
    # mu['SpleenTrans'] = [190.321392, 86.738878]
    # std['SpleenTrans'] = [41.459823, 21.711744]

    pts = Variable(pts.cuda())
    # for i in input

    # grid = np.meshgrid(range(input.size()[2]), range(input.size()[3]), indexing='ij')
    x0, y0 = weighted_center(input[0, 0, :, :])
    x1, y1 = weighted_center(input[0, 1, :, :])

    dist = torch.sqrt(torch.pow(x0-x1, 2)+torch.pow(y0-y1, 2))
    truedist = torch.sqrt(torch.pow(pts[0,0,0]-pts[0,1,0], 2)+torch.pow(pts[0,0,1]-pts[0,1,1], 2))
    loss = torch.abs(dist-truedist)
    #
    if category_name == 'KidneyTrans' or category_name == 'SpleenTrans':
    #     # x2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # y2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # x3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 3, :, :].sum()
    #     # y3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 3, :, :].sum()

        # dist2 = torch.sqrt(torch.pow(x2 - x3, 2) + torch.pow(y2 - y3, 2))
        # loss += torch.abs(dist2-mu[category_name][1])
        x2, y2 = weighted_center(input[0, 2, :, :])
        x3, y3 = weighted_center(input[0, 3, :, :])
        dist = torch.sqrt(torch.pow(x2-x3, 2)+torch.pow(y2-y3, 2))
        truedist = torch.sqrt(torch.pow(pts[0,2,0]-pts[0,3,0], 2)+torch.pow(pts[0,2,1]-pts[0,3,1], 2))
        loss += torch.abs(dist-truedist)
    # # criterion = torch.nn.L1Loss().cuda()
    # # loss = criterion(dist,mu[category_name][0])

    return loss

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)

    _, target_ = target.max(1)

    result_ = torch.squeeze(result_)
    target_ = torch.squeeze(target_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target = torch.cuda.FloatTensor(target_.size())
    else:
        result = torch.FloatTensor(result_.size())
        target = torch.FloatTensor(target_.size())
    result.copy_(result_.data)
    target.copy_(target_.data)
    result = result.view(-1)
    target = target.view(-1)
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU

def dice_loss_3d(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
    target = target.view(target.size(0), target.size(1), target.size(2), -1)
    input = input.view(input.size(0), input.size(1), input.size(2), -1)
    probs = F.softmax(input)

    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    dice_total = dice_total
    return dice_total





def dice_l2(input,target,size_average=True):
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(input, target)
    if size_average:
        loss /= (target.size()[0]*target.size()[1])
    return loss


def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.8):
    plt.ion()
    filters = units.shape[0]
    fig = plt.figure(figure_id, figsize=(5,5))
    fig.clf()

    for i in range(filters):
        plt.imshow(input_im[i,:,:], interpolation=interp, cmap='gray')
        plt.imshow(units[i,:,:], interpolation=interp, cmap=colormap, alpha=alpha)
        plt.axis('off')
        plt.colorbar()
        plt.title(title, fontsize='small')
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()



class Trainer(object):

    def __init__(self, cuda, model, optimizer=None,
                train_loader=None,test_loader=None,lmk_num=None,
                train_root_dir=None,out=None, max_epoch=None, batch_size=None,
                size_average=False, interval_validate=None,dual_network = False,
                add_calcium_mask=False,use_siamese = False,siamese_coeiff = 0.001):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.interval_validate = interval_validate

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        self.train_root_dir = train_root_dir
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.lmk_num = lmk_num
        self.siamese_coeiff = siamese_coeiff

        self.max_epoch = max_epoch
        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.batch_size = batch_size
        self.dual_network = dual_network
        self.add_calcium_mask = add_calcium_mask
        self.use_siamese = use_siamese

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.model, layer_name, upscale)
        return feature_extractor.forward(self.input)


    def validate(self,test_epoch=False):
        self.model.train()
        if test_epoch:
            out = osp.join(self.out, 'test')
        else:
            out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'test_accurarcy.txt')
        fv = open(log_file, 'a')
        log_file2 = osp.join(out, 'test_accurarcy_perepoch.txt')
        fv2 = open(log_file2, 'a')
        log_file3 = osp.join(out, 'test_recall_f1_acc_perepoch.txt')
        fv3 = open(log_file3, 'a')
        correct = 0
        correct_binary = 0

        pred_history=[]
        target_history=[]
        loss_history=[]
        sofar = 0


        for batch_idx, (data,target,sub_name) in tqdm.tqdm(
                # enumerate(self.test_loader), total=len(self.test_loader),
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            data, target = Variable(data,volatile=True), Variable(target,volatile=True)

            if self.dual_network:
                if self.add_calcium_mask:
                    data = data[:,0:2,: :,:]
                else:
                    data = data[:,0,:,:,:]
                    data = torch.unsqueeze(data, 1)
            pred_prob = self.model(data)

            if test_epoch:
                #get attention
                gcam = GradCAM(model=self.model)
                probs, idx = gcam.forward(data)
                topk = 3
                target_layer = 'ec6.2'
                # target_layer = 'ec1.2'
                test_attention_out = osp.join(out, target_layer)
                mkdir(test_attention_out)

                input_img = data[0, 0].data.cpu().numpy()
                input_size = (input_img.shape[0], input_img.shape[1], input_img.shape[2])
                input_mask = data[0, 1].data.cpu().numpy()
                nii_img = nib.Nifti1Image(input_img, affine=np.eye(4))
                output_img_file = os.path.join(out, ('%s_img.nii.gz' % sub_name[0]))
                nib.save(nii_img, output_img_file)
                nii_mask = nib.Nifti1Image(input_mask, affine=np.eye(4))
                output_mask_file = os.path.join(out, ('%s_mask.nii.gz' % sub_name[0]))
                nib.save(nii_mask, output_mask_file)
                del input_img,input_mask
                del nii_img, nii_mask

                for i in range(0, topk):
                    gcam.backward(idx=idx[i])
                    output = gcam.generate(target_layer=target_layer)
                    output = resize(output,input_size , mode='constant', preserve_range=True)

                    nii_seg = nib.Nifti1Image(output, affine=np.eye(4))
                    output_att_file = os.path.join(test_attention_out, ('%s_test_att%d_clss%d.nii.gz' % (sub_name[0],i,idx[i])))
                    nib.save(nii_seg, output_att_file)
                gcam.backward_del(idx=idx[i])
                del gcam, output, nii_seg, probs


                #training attention
                subnum = data.size(0)
                for subi in range(subnum):
                    attentions = []
                    i = 1
                    self.input = data
                    fmap = self.get_feature_maps('compatibility_score%d' % i, upscale=False)
                    try:
                        attmap = fmap[1][1]
                    except:
                        aaaa = 1
                    attention = attmap[subi,0].cpu().numpy()
                    # attention = attention[:, :]
                    # attention = numpy.expand_dims(resize(attention, (fmap_size[0], fmap_size[1]), mode='constant', preserve_range=True), axis=2)
                    attention = resize(attention, input_size, mode='constant', preserve_range=True)
                    attention = (attention-np.min(attention))/(np.max(attention)-np.min(attention))
                    # this one is useless
                    # plotNNFilter(fmap_0, figure_id=i+3, interp='bilinear', colormap=cm.jet, title='compat. feature %d' %i)

                    nii_seg = nib.Nifti1Image(attention, affine=np.eye(4))
                    output_att_file = os.path.join(out, ('%s_train_att.nii.gz' % sub_name[subi]))
                    nib.save(nii_seg, output_att_file)
                    del nii_seg, fmap, attmap, attention

                # plotNNFilterOverlay(input_img, attention, figure_id=i, interp='bilinear', colormap=cm.jet,
                #                     title='a', alpha=0.5)
                # attentions.append(attention)


            pred_clss = F.log_softmax(pred_prob)
            pred = pred_clss.data.max(1)[1]  # get th
            correct += pred.eq(target.data).cpu().sum()

            pred_binary = pred>0
            target_binary = target.data>0
            correct_binary += pred_binary.eq(target_binary).cpu().sum()

            sofar += data.size(0)

            test_loss = F.nll_loss(pred_clss, target)

            for batch_num in range(data.size(0)):
            # test_loss /= len(self.test_loader)  # loss function already averages over batch size
                results_strs = '[Epoch %04d] True=[%d],Pred=[%d],Pred_prob=%s,Test set: Average loss: %.4f, Accuracy: %d/%d (%.3f) binary (%.3f), subname=[%s]\n' % (
                    self.epoch, target.data.cpu().numpy()[batch_num], pred.cpu().numpy()[batch_num], np.array2string(pred_clss[batch_num].data.cpu().numpy()), test_loss.data[0], correct, sofar,
                    100. * float(correct) / sofar, 100 * float(correct_binary) / sofar, sub_name[batch_num])
                print(results_strs)
                fv.write(results_strs)

            loss_history.append(test_loss.data.cpu().numpy().tolist())
            pred_history += pred_binary.cpu().numpy().tolist()
            target_history += target_binary.data.cpu().numpy().tolist()

        f1 = f1_score(target_history, pred_history)
        recall = recall_score(target_history, pred_history)
        precision = precision_score(target_history, pred_history)
        accuracy = accuracy_score(target_history, pred_history)

        print_str='test epoch='+str(self.epoch)+',accuracy='+str(accuracy)+",f1="+str(f1)+",recall="+str(recall)+',precision='+str(precision)+",loss="+str(np.mean(loss_history))+"\n"

        fv2.write(results_strs)
        fv3.write(print_str)
        fv.close()
        fv2.close()
        fv3.close()

    def train(self):
        self.model.train()
        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')
        log_file2 = osp.join(out, 'training_loss_perepoch.txt')
        fv2 = open(log_file2, 'a')

        correct = 0
        correct_binary = 0
        sofar = 0
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
  
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.optim.zero_grad()
            if self.dual_network:
                if self.add_calcium_mask:
                    data2 = data[:,2:4,:,:,:]
                    data1 = data[:,0:2,: :,:]
                    pred_prob2 = self.model(data2)
                    pred_clss2 = F.log_softmax(pred_prob2)
                    pred_prob = self.model(data1)
                    pred_clss = F.log_softmax(pred_prob)
                else:
                    data2 = data[:,1,:,:,:]
                    data1 = data[:,0,:,:,:]
                    pred_prob2 = self.model(torch.unsqueeze(data2,1))
                    pred_clss2 = F.log_softmax(pred_prob2)
                    pred_prob = self.model(torch.unsqueeze(data1,1))
                    pred_clss = F.log_softmax(pred_prob)
                loss1 = F.nll_loss(pred_clss, target)
                loss2 = F.nll_loss(pred_clss2, target)
                if self.use_siamese:
                    criterion_siamese = ContrastiveLoss()
                    # loss3 = l1_loss(pred_clss,pred_clss2)
                    loss3 = criterion_siamese(pred_prob2,pred_prob)
                    loss = loss1+loss2+self.siamese_coeiff*loss3
                else:
                    loss3= loss2
                    loss = loss1+loss2
                # loss = loss1 + loss2 + 0.1 * loss3
            else:
                pred_prob = self.model(data)
                pred_clss = F.log_softmax(pred_prob)
                loss = F.nll_loss(pred_clss, target)

            # #see cam
            # features_blobs = []
            #
            # def hook_feature(module, input, output):
            #     features_blobs.append(output.data.cpu(loss3).numpy())
            #
            # finalconv_name = 'features'
            # self.model._modules.get(finalconv_name).register_forward_hook(features_blobs)
            # params = list(self.model.parameters())
            # weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            #
            # def returnCAM(feature_conv, weight_softmax, class_idx):
            #     # generate the class activation maps upsample to 256x256
            #     size_upsample = (256, 256)
            #     bz, nc, h, w = feature_conv.shape
            #     output_cam = []
            #     for idx in class_idx:
            #         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            #         cam = cam.reshape(h, w)
            #         cam = cam - np.min(cam)
            #         cam_img = cam / np.max(cam)
            #         cam_img = np.uint8(255 * cam_img)
            #         output_cam.append(cv2.resize(cam_img, size_upsample))
            #
            #     return output_cam
            #
            # h_x = F.softmax(pred_prob).data.squeeze()
            # probs, idx = h_x.sort(0, True)
            # probs = probs.numpy()
            # idx = idx.numpy()
            # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])


            sofar += data.size(0)

            pred = pred_clss.data.max(1)[1]  # get th
            correct += pred.eq(target.data).cpu().sum()

            pred_binary = pred>0
            target_binary = target.data>0
            correct_binary += pred_binary.eq(target_binary).cpu().sum()
            total = (batch_idx+1)*self.batch_size

            if (batch_idx % 1 == 0):
                if self.dual_network:
                    print_str = 'epoch=%d, batch_idx=%d, loss=%.4f+%.4f +%.4f= %.4f, Accuracy: %d/%d (%.3f) binary (%.3f)\n' % (
                        self.epoch, batch_idx, loss1.data[0], loss2.data[0], loss3.data[0], loss.data[0],correct, sofar,
                        100. * float(correct) / sofar, 100 * float(correct_binary) / sofar)
                else:
                    print_str = 'epoch=%d, batch_idx=%d, loss=%.4f, Accuracy: %d/%d (%.3f) binary (%.3f)\n' % (self.epoch, batch_idx, loss.data[0], correct, sofar,
                        100. * float(correct) / sofar, 100 * float(correct_binary) / sofar)
                print(print_str)
                fv.write(print_str)
            loss.backward()
            self.optim.step()
        fv2.write(print_str)
        fv.close()
        fv2.close()

    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models')
            mkdir(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)


            if os.path.exists(model_pth):
                if self.cuda:
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                # self.model.load_state_dict(torch.load(model_pth))
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
                # if epoch % 5 == 0:
                self.validate()
            else:
                # self.validate()
                self.train()
                # if epoch % 5 == 0:
                self.validate()
                torch.save(self.model.state_dict(), model_pth)

                # torch.save(self.model.state_dict(), model_pth)

    def test_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Test', ncols=80):
            self.epoch = epoch
            train_root_dir = osp.join(self.out, 'models')

            model_pth = '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
            if os.path.exists(model_pth):
                if self.cuda:
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    # self.model.load_state_dict(torch.load(model_pth))
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            self.validate(test_epoch=True)
            return

             



