import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from dataloaders.la_heart import _label_decomp

def getLargestCC_multiclass(segmentation, num_classes):
    largestCC_list = []
    for i in range(num_classes):
        labels = label(segmentation[i,:,:,:])
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        largestCC_list.append(largestCC)
    return np.array(largestCC_list)

def getLargestCC(segmentation, num_classes):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    total_metric = 0.0
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    multi_class = False
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label = _label_decomp(label, num_classes)
        label = label.argmax(axis=0)

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes)
        prediction = score_map
        prediction =  prediction.argmax(axis=0)
        if nms:
            prediction = getLargestCC(prediction, num_classes)
        
        single_metric = calculate_metric_percase(prediction, label, num_classes)
        
       
        if metric_detail:
            if multi_class == True:
                for i in range(num_classes+1):
                    print('%02d, %02d, \t%.5f, %.5f, %.5f, %.5f' % (
                        ith, i, single_metric[0][i], single_metric[1][i], single_metric[2][i], single_metric[3][i]))
                total_metric += np.asarray(single_metric)[:,num_classes]
            else:
                print('%02d, \t%.5f, %.5f, %.5f, %.5f' % (
                        ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
                total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32),
                                     np.eye(4)), test_save_path + "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(
                4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(
                4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros((num_classes, ) + image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, x1, x2, x3, x4, x5, _, _, _, _ = net(test_patch)
                    # ensemble
                    y = torch.sigmoid(y1)
                y = y.cpu().data.numpy()

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                        = cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/cnt
    label_map = (score_map > 0.5).astype(np.int)

    if add_pad:
        label_map = label_map[:, wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)
        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice
    return total_dice


def calculate_metric_percase_multiclass(pred, gt, num_classes):
    dice = []
    jc = []
    hd = []
    asd = []
    for i in range(num_classes):
        dice.append(metric.binary.dc(pred[i], gt[i]))
        jc.append(metric.binary.jc(pred[i], gt[i]))
        hd.append(metric.binary.hd95(pred[i], gt[i]))
        asd.append(metric.binary.asd(pred[i], gt[i]))
    dice.append(np.mean(dice))
    jc.append(np.mean(jc))
    hd.append(np.mean(hd))
    asd.append(np.mean(asd))

    return dice, jc, hd, asd

def calculate_metric_percase(pred, gt, num_classes):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd
