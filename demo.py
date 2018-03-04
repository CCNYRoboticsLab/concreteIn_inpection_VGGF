import numpy as np
import theano
import theano.tensor as T
import lasagne


#import skimage.transform
import pickle
import os

import re
import Image


##build the vgg model

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7_2048'] = DenseLayer(net['fc6'], num_units=2048)
    net['fc8_7'] = DenseLayer(net['fc7_2048'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8_7'], softmax)

    return net


#with np.load('/home/robolab/project/deepProjectBridge/YL_50iter.npz') as f:
#    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#    print(range(len(param_values)))
#print len(param_values)

BATCH_SIZE = 1


net = build_model()

#half_feature_layer=DenseLayer(net['fc6'], num_units=2048)
#half_feature_layer_dp=DropoutLayer(half_feature_layer,p=0.)
#output_layer=DenseLayer(half_feature_layer_dp,num_units=2,nonlinearity=softmax)
#final_prob=NonlinearityLayer(output_layer, softmax)

#lasagne.layers.set_all_param_values(net['fc7_2048'], param_values[:30])

half_feature_layer=DenseLayer(net['fc6'],num_units=2048)
half_feature_layer_dp=DropoutLayer(half_feature_layer,p=0.)
output_layer=DenseLayer(half_feature_layer_dp,num_units=2,nonlinearity=softmax)
final_prob=NonlinearityLayer(output_layer, softmax)


with np.load('/home/robolab/project/deepProjectBridge/YL_50iterV2.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output_layer, param_values)


print 'successfully..., from new model 500'

###############################################################
# begin to load images and do the test
# Define loss function and metrics, and get an updates dictionary

# 11111111111111111
# define the function and cost stuff

X_sym = T.tensor4()
y_sym = T.ivector()

prediction = lasagne.layers.get_output(final_prob, X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                      dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(final_prob, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)


# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)

##################################################################

# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk)==N:
            rst=chunk
            chunk=[]
            yield rst
    if chunk:
        yield chunk



#####################################################################
# defines functions for pre-process the image data

def train_batch():
    trdata,trlb=imdata(imglist)
    trdata/=256
    #trdata=trdata-MEAN_IMAGE
    return train_fn(trdata,trlb)

def test_batch():
    tsdata,tslb=imdata(ixx)
    tsdata/=256
    #tsdata=tsdata-MEAN_IMAGE
    return val_fn(tsdata,tslb)

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])



#####################################################################


# print trdata[0:5,:,:,:], trlb

##things to change

# print 'begin training'
    
loss_tot = 0.
acc_tot = 0.

# loss,acc=test_batch()
# print 'Test loss and acc:',loss,acc
import math
import cv2
import random

print 'direct testing'

regionSize = 200;

for kk in range(1):
	img=cv2.imread('bridgeFlaws/%03d.jpg'%(232))
        #print('bridgeFlaws/%3d.jpg',kk+1)
        #img=cv2.imread('imageSource/'+s)
	height, width, channels = img.shape
	widthLevel = int(width/regionSize);
	heightLevel = int(height/regionSize);
	imgShow = np.zeros((height,width,3))
	imgShow = np.copy(img)
	for ii in range(widthLevel-1):
	     for jj in range(heightLevel-1):
		   #xRange = width - 304
		   #yRange = height - 304
		   #randXCut = random.random()
		   #randYCut = random.random()
		   #rangeXForCut = [1+ math.floor(xRange*randXCut), math.floor(xRange*randXCut) + 300]
		   #rangeYForCut = [1+ math.floor(yRange*randYCut), math.floor(yRange*randYCut) + 300]
		   rangeXForCut = [(ii-1)*regionSize +1, (ii-1)*regionSize + regionSize*2]
		   rangeYForCut = [(jj-1)*regionSize +1, (jj-1)*regionSize + regionSize*2]
		   print(int(rangeXForCut[0]), int(rangeXForCut[1]), int(rangeYForCut[0]), int(rangeYForCut[1]))
		   #tempImg = img[int(rangeXForCut[0]):int(rangeXForCut[1]), 100:200, :]
		   tempImg = img[int(rangeYForCut[0]):int(rangeYForCut[1]), int(rangeXForCut[0]):int(rangeXForCut[1]), :]

		   height1, width1, channels1 = tempImg.shape

		   if height1 != 0 and width1 != 0:
			   im224=np.zeros((3,224,224))
			   for t in range(3):
			       im224[t,:,:]=cv2.resize(tempImg[:,:,t],(224,224))
			   datablob=np.ndarray((1,3,224,224))
			   datalb=np.zeros((1,))
			   datablob[0,:,:,:]=im224
			   datalb[0]=int(1.0)
			   datablob=datablob.astype('float32')
			   datablob/=256
			   datalb=datalb.astype('int32')
			   loss, acc = val_fn(datablob, datalb)
			   prediction = pred_fn(datablob)
			   #print('prediction')
			   #print prediction[0][0]
			   #print datalb
			   if prediction[0][0] < 0.5:
			      cv2.rectangle(imgShow,(int(rangeXForCut[1]), int(rangeYForCut[1])), (int(rangeXForCut[0]), int(rangeYForCut[0])), (0,255,0),3)
			   #else:
			   #  cv2.rectangle(imgShow,(int(rangeXForCut[1]), int(rangeYForCut[1])), (int(rangeXForCut[0]), int(rangeYForCut[0])), (255,0,0),1)
	cv2.imwrite('generated/'+str(232)+'.jpg',imgShow)
#####################################################################

