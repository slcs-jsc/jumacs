"""Independent evaluation climatology versus a model, without adjustment."""
import numpy as np
import xarray as xr
from .comparison import compare_fields


def evaluate_fields(model_field, evaluation_field, model_pressure, evaluation_pressure, target_pa):
    compared = compare_fields(model_field, evaluation_field, model_pressure,
                              evaluation_pressure, target_pa, ("model", "evaluation"))
    compared["rms_difference"] = np.sqrt((compared["difference"] ** 2).mean(("lat", "pressure"), skipna=True))
    compared.attrs["purpose"] = "evaluation only; no adjustment, scoring, or model ranking"
    return compared
