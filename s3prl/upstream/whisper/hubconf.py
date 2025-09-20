# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def whisper_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def whisper_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return whisper_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def whisper_tiny(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"

    return whisper_url(refresh=refresh, *args, **kwargs)

def whisper_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"

    return whisper_url(refresh=refresh, *args, **kwargs)

def whisper_small(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"

    return whisper_url(refresh=refresh, *args, **kwargs)

def whisper_medium(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"

    return whisper_url(refresh=refresh, *args, **kwargs)

def whisper_large(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"

    return whisper_url(refresh=refresh, *args, **kwargs)
