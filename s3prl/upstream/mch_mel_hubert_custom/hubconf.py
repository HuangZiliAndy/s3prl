from .expert import UpstreamExpert as _UpstreamExpert


def mch_mel_hubert_custom_local(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
