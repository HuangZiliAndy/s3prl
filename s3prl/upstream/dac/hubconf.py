# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/hubconf.py ]
#   Synopsis     [ the mockingjay torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert
import dac

def dac_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
        feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def dac_16kHz(**kwargs):
    kwargs["ckpt"] = dac.utils.download(model_type="16khz")
    kwargs["quantize"] = True 
    return _UpstreamExpert(**kwargs)

def dac_16kHz_no_quantize(**kwargs):
    kwargs["ckpt"] = dac.utils.download(model_type="16khz")
    kwargs["quantize"] = False 
    return _UpstreamExpert(**kwargs)
