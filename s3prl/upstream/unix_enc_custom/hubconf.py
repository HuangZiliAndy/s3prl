from .expert import UpstreamExpert as _UpstreamExpert


def unix_enc_custom_local(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
