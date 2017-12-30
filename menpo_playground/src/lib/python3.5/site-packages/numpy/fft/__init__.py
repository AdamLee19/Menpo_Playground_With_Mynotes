from __future__ import division, absolute_import, print_function

# To get sub-modules
from .info import __doc__

try:
    from mklfft.fftpack import *
    using_mklfft = True
except ImportError:
    from .fftpack import *
    using_mklfft = False
from .helper import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
