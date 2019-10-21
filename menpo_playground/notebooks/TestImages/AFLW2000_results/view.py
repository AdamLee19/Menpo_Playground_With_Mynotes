import numpy as np
import menpo3d.io as mio

mesh = mio.import_builtin_asset('james.obj')
viewer = mesh.view()