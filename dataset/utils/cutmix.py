import numpy as np
import torch

def obtain_cutmix_box(img_size, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.ones(img_size, img_size)
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 0

    return mask

def obtain_bbox(batch_size, img_size):
    for i in range(batch_size):  
        if i == 0:
            MixMask = obtain_cutmix_box(img_size).unsqueeze(0)
        else:
            MixMask = torch.cat((MixMask, obtain_cutmix_box(img_size).unsqueeze(0)))
    return MixMask

def mix(mask, data_l, data_ul, rand_index=None):
    # print('mask.shape, data_l.shape, data_ul.shape:',mask.shape, data_l.shape, data_ul.shape)
    # get the random mixing objects
    if rand_index is None:
        rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    #Mix
    # data = torch.cat([(mask[i] * data_ul[rand_index[i]] + (1 - mask[i]) * data_l[i]).unsqueeze(0) for i in range(data_l.shape[0])])
    data_tmp = []
    for i in range(data_l.shape[0]):
        tmp = (mask[i] * data_ul[rand_index[i]] + (1 - mask[i]) * data_l[i]).unsqueeze(0) 
        data_tmp.append(tmp)
    data = torch.cat(data_tmp)
    
    return data,rand_index