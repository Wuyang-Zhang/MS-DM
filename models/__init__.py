"""Public model entry point.

``vgg19`` remains the compatibility name used by the training and prediction
code and resolves to the final MS-DM implementation in :mod:`models.msdm`.
The original DM-Count-style baseline is available from :mod:`models.dm_count`.
"""

from .msdm import vgg19

__all__ = ["vgg19"]
