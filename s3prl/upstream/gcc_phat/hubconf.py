import os

from .expert import UpstreamExpert as _UpstreamExpert


def gcc_phat(model_config, *args, **kwargs):
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)
