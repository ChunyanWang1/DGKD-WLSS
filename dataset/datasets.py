import torch
from torch.utils import data
import os.path as osp
import numpy as np
import random
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
import os
from torchvision import transforms
#import torchvision.transforms as standard_transforms

import imageio
from torchvision import transforms as T

# import sys
# # # 获取当前工作目录的路径
# current_dir = os.getcwd()
# sys.path.append(current_dir)
from . import imutils

IMG_FOLDER_NAME = 'JPEGImages'
IMG_FOLDER_NAME2 = "darkened_VOC2012" 
IMG_FOLDER_NAME3='vis_depth'
MASK_FOLDER_NAME = 'SegmentationClassAug'
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('/opt/data/private/diffkd_segmentation/dataset/voc12/cls_labels.npy', allow_pickle=True).item()

def denorm(ori_images):
    #ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, 0] = (ori_images[:, :, 0] * 0.229 + 0.485)*255
    ori_images[:, :, 1] = (ori_images[:, :, 1] * 0.224 + 0.456)*255
    ori_images[:, :, 2] = (ori_images[:, :, 2] * 0.225 + 0.406)*255

    return ori_images.astype(np.uint8)#.permute(0,3,1,2).contiguous()

def norm(ori_images):
    #ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[ :, :, 0] = (ori_images[ :, :, 0] - 0.485)/ 0.229
    ori_images[ :, :, 1] = (ori_images[ :, :, 1] - 0.456)/ 0.224
    ori_images[ :, :, 2] = (ori_images[ :, :, 2] - 0.406)/ 0.225

    return ori_images#.permute(0,3,1,2).contiguous()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]


def encode_str_filename(str_filenames):
    s = int(str_filenames[:4] + str_filenames[5:])
    return s

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def get_label_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, MASK_FOLDER_NAME, img_name + '.png')

def get_img_path2(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME2, img_name + '.jpg')

def get_img_path3(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME3, img_name + '_depth.png') 


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [encode_str_filename(img_gt_name) for img_gt_name in img_gt_name_list]
    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    # def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    #     self.mean = mean
    #     self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ImageDataset(data.Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
    #def __init__(self,voc12_root, img_name_list_path, transform=None):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.num_class = 21

       

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str =decode_int_filename(name)
        #print(name_str)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        d_img=np.asarray(imageio.imread(get_img_path2(name_str, self.voc12_root)))
        depth_img=np.asarray(imageio.imread(get_img_path3(name_str, self.voc12_root)))
        mask=np.asarray(imageio.imread(get_label_path(name_str, self.voc12_root)))
        if self.resize_long:
            img,d_img,depth_img,mask = imutils.random_resize_long3(img,d_img,depth_img,mask, self.resize_long[0], self.resize_long[1])
        if self.rescale:
            img,d_img,depth_img,mask = imutils.random_scale3(img,d_img,depth_img,mask, scale_range=self.rescale, order=3)
        #
        
        if self.img_normal:
            img = self.img_normal(img)
            d_img = self.img_normal(d_img)
            depth_img = self.img_normal(depth_img)
        

        if self.hor_flip:
            img,d_img,depth_img,mask = imutils.random_lr_flip3(img,d_img,depth_img,mask)

        if self.crop_size:
            if self.crop_method == "random":
                img,d_img,depth_img,mask = imutils.random_crop3(img,d_img,depth_img,mask, self.crop_size, 0)
            else:
                img,d_img,depth_img,mask = imutils.top_left_crop3(img,d_img,depth_img,mask, self.crop_size, 0)


        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
            d_img = imutils.HWC_to_CHW(d_img)
            depth_img=imutils.HWC_to_CHW(depth_img)

        return {'name': name_str, 'img': img,'d_img':d_img,'depth_img':depth_img,'target':mask}#  'depth_img':depth_img,
        
        #return name_str, img, e_img#'hist_img':equalized_img

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out['img'],out['d_img'],out['depth_img'],out['target'],out['label'],out['name'] #out['hist_img'] out['depth_img']



class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales
        self.transforms=transforms.Resize((512,512))#(448,448)(512,512)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        d_img=imageio.imread(get_img_path2(name_str, self.voc12_root))
        depth_img=imageio.imread(get_img_path3(name_str, self.voc12_root))
        mask=imageio.imread(get_label_path(name_str, self.voc12_root))
    
        ms_img_list = []
        d_ms_img_list=[]
        depth_ms_img_list=[]
        hist_ms_img_list=[]
        mask_ms_list=[]
        s_mask=imutils.pil_rescale(mask, 1, order=0)
        for s in self.scales:
            if s == 1:
                s_img = img
                s_d_img=d_img
                s_depth_img=depth_img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
                s_d_img = imutils.pil_rescale(d_img, s, order=3)
                s_depth_img = imutils.pil_rescale(depth_img, s, order=3)
                

            s_img = self.img_normal(s_img)
            s_d_img=self.img_normal(s_d_img)
            s_depth_img=self.img_normal(s_depth_img)
            

            s_img = imutils.HWC_to_CHW(s_img)
           
            s_d_img=imutils.HWC_to_CHW(s_d_img)

            s_depth_img=imutils.HWC_to_CHW(s_depth_img)

            
            ms_img_list.append(s_img)  
            depth_ms_img_list.append(s_depth_img)
            d_ms_img_list.append(s_d_img) 

        out = {"name": name_str, "img": ms_img_list,"d_img": d_ms_img_list,"depth_img":depth_ms_img_list, "target":s_mask, #"size": (img.shape[0], img.shape[1]),#"hist_img": hist_ms_img_list
               "label": torch.from_numpy(self.label_list[idx])}
        return out['img'],out['d_img'],out['depth_img'],out['target'],out['label'], out['name']
