from menpowidgets import visualize_images as vimgs
from menpo.transform import scale_about_centre
import numpy as np 
from menpo.image import Image
import os, shutil, sys
import menpo.io as mio
from menpowidgets import visualize_images
import numpy as np
import matplotlib.pyplot as plt
import menpo.math.decomposition as mmath



'''
Defined function
'''

#landmakr handler
def find_landmark(asset):
    index = str(asset).rfind('/') + 1
    landmark + str(asset)[ index: -4 ]
    return { "MyAAM": os.path.abspath( landmark + '/' + str(asset)[ index: -4 ] + '.pts' ) }

def process_bar( now, max, prompt ):
	percentage = now / max * 100.0
	sys.stdout.write( "\r{}: {:.2f}%".format( prompt, percentage ) )
	sys.stdout.flush()

def weight_compare( W_train, W_test ):
	return np.linalg.norm( W_train - W_test )





'''
Global variable
'''
#working directory
workingspace = os.getcwd()
trainingdata = workingspace + '/1/'
landmark = workingspace + '/2/'
testdata = workingspace + '/test/'
meanfaceDir = workingspace + '/mean.png'
weight = workingspace + '/weight/'
threshold = 100000



#Initialize a blank 512*512 image
trainingset = Image.init_blank( [ 512, 512 ] )
#image vectors
imgVs = []
#image name and image vector pair
dictionary = {}
#Training vector
TV = trainingset.as_vector()




#How many images
count = 0
for line in os.listdir( trainingdata ):
    if line.endswith( '.jpg' ):
        count += 1




'''
Read images & calculate mean face
'''
now = 0.0
prompt = "Reading Images"
for fname in os.listdir( trainingdata ):  
    if( fname.endswith( '.jpg' ) ):
        now += 1.0
        process_bar( now, count, prompt )
        #landmark, greyscale, crop, resize, vectorize
        img = mio.import_image( trainingdata + fname , landmark_resolver = find_landmark )
        if img.n_channels == 3:
            img = img.as_greyscale()
        img = img.crop_to_landmarks().resize([512,512])
        imgV = img.as_vector()
        dictionary[ fname ] = imgV
        TV = TV + imgV
#Get mean face and export                
meanV = TV / count
meanFace = trainingset.from_vector( meanV )
print( "\nExport mean face" )
if( os.path.exists( meanfaceDir ) ):
    os.remove( meanfaceDir )
    mio.export_image( meanFace, meanfaceDir )
else:
    mio.export_image( meanFace, meanfaceDir )





'''
Get eigenvector
'''

A = np.empty( (512*512, count ) )
columni = 0
prompt = "\rEigenvector"
for key in dictionary:
    process_bar( columni + 1, count, prompt )
    dictionary[ key ] = dictionary[ key ] - meanV
    A[ :, columni ] = dictionary[ key ]
    columni += 1
C = np.dot(A.transpose(), A) / ( count - 1 )
eigVector, eigValue = mmath.eigenvalue_decomposition( C )
U_k = np.dot( A, eigVector )
#Export eigenvector
#print( "\n{}".format(U_k.shape) )
# fp_uk = open( weight + 'U_k.txt', 'w+' )
# c = 0
# fp_uk.write( str(U_k.shape) + '\n' )
# i = 0
# j = 0
# prompt = "\rExport EigenVector"
# while( i < U_k.shape[ 0 ] ):
# 	while( j < U_k.shape[ 1 ] ):
# 		process_bar( i*U_k.shape[ 1 ] + j + 1, U_k.shape[ 0 ]*U_k.shape[ 1 ], prompt)
# 		fp_uk.write( str( U_k[ i ][ j ] ) + ' ' )
# 		j += 1
# 	i += 1
# 	fp_uk.write( '\n' )
# 	j = 0
# fp_uk.close()


'''
Export weight
Cheacking and creating a directory for weight
'''
# if( os.path.exists( weight ) ):
#     shutil.rmtree( weight )
#     os.mkdir( weight )
# else:
#     os.mkdir( weight )

print()
now = 0
prompt = "\rExport weight"
for key in dictionary:
    now += 1
    process_bar( now, count, prompt )
    W_k = np.dot( U_k.transpose(), dictionary[ key ] )
    #print ("fdfsdfs{}".format(dictionary[ key ].shape))
    dictionary[ key ] = W_k
    #print (dictionary[ key ].shape)
    # fp_weight = open( weight + key[:-4] + '.txt', 'w+')
    # for w in W_k:
    #     fp_weight.write("{}\n".format(w))
    # fp_weight.close()


result = {}
now = 0
prompt = "\rCompare face"
min = 999999
global pair 
#pair = ( ,)
for img in os.listdir( testdata ):
	if img.endswith( 'jpg' ):
		print( "\nfdfsdad\n" + img )
		obama = mio.import_image( testdata + img  )
		if obama.n_channels == 3:
			obama = obama.as_greyscale()
		obama = obama.resize([512,512])
		obamaV = obama.as_vector() - meanV
		w_obama = np.dot( U_k.transpose(), obamaV )
		for key in dictionary:
			now += 1
			process_bar( now, count, prompt )
			r = weight_compare( dictionary[ key ], w_obama )
			if( r < min ):
				min = r
				pair = (key,min)
			if( r < threshold ):
				result[ key ] = r


print( "\n{}".format(result) )
print()
print( pair )