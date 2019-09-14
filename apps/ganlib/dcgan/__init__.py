"""
The :mod:`ganlib.dcgan` module includes the DCGAN architecture.

"""

from .dcgan import DCGAN
from .dcgan import Generator
from .dcgan import Discriminator


__all__ = [
    'Generator',
    'Discriminator',
    'DCGAN',
]