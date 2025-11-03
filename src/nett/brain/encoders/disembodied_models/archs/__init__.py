# Mute excessive warnings from pl_bolts, see https://github.com/Lightning-Universe/lightning-bolts/issues/563

import warnings

warnings.filterwarnings(
    "ignore", message=r"The feature ([^\s]+) is currently marked under review"
)

original_filterwarnings = warnings.filterwarnings


def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, "append": True})


warnings.filterwarnings = _filterwarnings
