#%matplotlib inline
import numpy as np
import menpo.io as mio
from menpo.image import Image




# Import image. Print image type, instance, pixels shape, n_channels, shape
lenna = mio.import_builtin_asset( 'lenna.png' )
print( 'Lenna is a {}'.format( type( lenna ) ) )
print( "Lenna 'is an' Image: {}".format( isinstance( lenna, Image) ) )
print( 'Lenna shape: {}'.format( lenna.pixels.shape ) )
print( 'The number of channels in Lenna is {}'.format( lenna.pixels.shape[0] ) )
print( 'n_channels for Lenna is {}'.format( lenna.n_channels) )
print( 'Lenna has a shape of {}'. format( lenna.shape ) )