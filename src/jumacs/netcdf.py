"""NetCDF opening helpers: one place to keep cftime time decoding current."""
from functools import lru_cache


def _coder_class():
    try:
        from xarray.coders import CFDatetimeCoder
    except ImportError:
        from xarray.coding.times import CFDatetimeCoder
    return CFDatetimeCoder


@lru_cache(maxsize=1)
def cftime_decoder():
    """The decoder xarray asks for in place of the deprecated ``use_cftime=True`` keyword."""
    return _coder_class()(use_cftime=True)


@lru_cache(maxsize=1)
def _coder_is_enough():
    """Whether a coder passed to ``decode_times`` still forces cftime timestamps."""
    try:
        import numpy as np
        import xarray as xr
        from xarray.core.variable import Variable

        days = Variable(("time",), np.array([0.0, 1.0]),
                        {"units": "days since 2000-01-01", "calendar": "standard"})
        probe = xr.Dataset(coords={"time": days})
        return xr.decode_cf(probe, decode_times=cftime_decoder()).time.dtype == object
    except Exception:
        return False


def open_cftime_dataset(path, **kwargs):
    """Open a dataset with cftime timestamps, which CCMI's non-standard calendars require."""
    import xarray as xr

    kwargs.pop("decode_times", None)
    if _coder_is_enough():
        kwargs.pop("use_cftime", None)
        return xr.open_dataset(path, decode_times=cftime_decoder(), **kwargs)
    kwargs["use_cftime"] = True
    return xr.open_dataset(path, decode_times=True, **kwargs)
