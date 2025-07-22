"""Model store which handles pretrained models """

from .pspnet import *
from .ssss import *
from .deeplabv3 import *
from .deeplabv3_mobile import *
from .psp_mobile import *
#from .deeplabv3plus import *
from .toco import *

__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    models = {
        'psp': get_psp,
        'ssss': get_ssss,
        'deeplabv3':get_deeplabv3,
        'deeplabv3plus':get_deeplabv3plus,
        'deeplab_mobile': get_deeplabv3_mobile,
        'psp_mobile': get_psp_mobile,
        'toco':get_toco,
    }
    return models[model](**kwargs)
