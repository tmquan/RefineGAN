#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob



# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation


# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

# Tensorflow 1
import tensorflow as tf
from tensorflow import layers
# from tensorflow.contrib.layers.python import layers
###############################################################################
SHAPE = 256
BATCH = 1
TEST_BATCH = 100
EPOCH_SIZE = 100
NB_FILTERS = 64  # channel size

DIMX  = 256
DIMY  = 256
DIMZ  = 2
DIMC  = 1
###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)
###############################################################################
def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)
###############################################################################
# Utility function for scaling 
def cvt2tanh(x, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / 255.0 - 0.5) * 2.0
###############################################################################
def cvt2imag(x, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * 255.0
###############################################################################		
def cvt2sigm(x, name='ToRangeSigm'):
	with tf.variable_scope(name):
		return (x / 1.0 + 1.0) / 2.0
###############################################################################
def tf_complex(data, name='tf_channel'):
	with tf.variable_scope(name+'_scope'):
		real  = data[:,0:1,...]
		imag  = data[:,1:2,...]
		del data
		data  = tf.complex(real, imag) 
	data = tf.identity(data, name=name)
	return data	
###############################################################################
def tf_channel(data, name='tf_complex'):
	with tf.variable_scope(name+'_scope'):
		real  = tf.real(data)
		imag  = tf.imag(data)
		real  = real[:,0:1,...]
		imag  = imag[:,0:1,...]
		del data
		data  = tf.concat([real, imag], axis=1)
	data = tf.identity(data, name=name)
	return data
###############################################################################
def np_complex(data):
	real  = data[0,...]
	imag  = data[1,...]
	del data
	data = real + 1j*imag
	return data	

###############################################################################
def np_channel(data):
	real  = np.real(data)
	imag  = np.imag(data)
	del data
	data  = np.concatenate([real, imag], axis=1)
	return data		

###############################################################################
# tfutils.symbolic_functions.psnr(prediction, ground_truth, maxp=None, name='psnr')
def psnr(prediction, ground_truth, maxp=None, name='psnr'):
	"""`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

	.. math::

		PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)

	Args:
		prediction: a :class:`tf.Tensor` representing the prediction signal.
		ground_truth: another :class:`tf.Tensor` with the same shape.
		maxp: maximum possible pixel value of the image (255 in in 8bit images)

	Returns:
		A scalar tensor representing the PSNR.
	"""
	prediction   = tf.abs(prediction)
	ground_truth = tf.abs(ground_truth)
	def log10(x):
		with tf.name_scope("log10"):
			numerator = tf.log(x)
			denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
			return numerator / denominator

	mse = tf.reduce_mean(tf.square(prediction - ground_truth))
	if maxp is None:
		psnr = tf.multiply(log10(mse), -10., name=name)
	else:
		maxp = float(maxp)
		psnr = tf.multiply(log10(mse+1e-6), -10.)
		psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)
	add_moving_summary(psnr)
	return psnr
			
###############################################################################
def RF(image, mask, name="RF"):
	# This op perform undersampling
	with tf.variable_scope(name+'_scope'):
		# Convert from 2 channel to complex number
		image = tf_complex(image)
		mask  = tf_complex(mask) 

		# Forward Fourier Transform
		freq_full = tf.fft2d(image, name='Ff')
		freq_zero = tf.zeros_like(freq_full)
		condition = tf.cast(tf.real(mask)>0.9, tf.bool)

		freq_dest = tf.where(condition, freq_full, freq_zero, name='RfFf')

		# Convert from complex number to 2 channel
		freq_dest = tf_channel(freq_dest)
	return tf.identity(freq_dest, name=name)

###############################################################################
def FhRh(freq, mask, name='FhRh', is_normalized=False):
	with tf.variable_scope(name+'_scope'):
		# Convert from 2 channel to complex number
		freq = tf_complex(freq)
		mask = tf_complex(mask) 

		# Under sample
		condition = tf.cast(tf.real(mask)>0.9, tf.bool)
		freq_full = freq
		freq_zero = tf.zeros_like(freq_full)
		freq_dest = tf.where(condition, freq_full, freq_zero, name='RfFf')

		# Inverse Fourier Transform
		image 	  = tf.ifft2d(freq_dest, name='FtRt')
		
		if is_normalized:
			image = tf.div(image, ((DIMX-1)*(DIMY-1)))

		# Convert from complex number to 2 channel
		image = tf_channel(image)
	return tf.identity(image, name)

###############################################################################
def update(recon, image, mask, name='update'):
	"""
	Update the reconstruction with undersample k-space measurement
	"""
	with tf.variable_scope(name+'_scope'):
		k_recon = RF(recon, tf.ones_like(mask), name='k_recon')
		k_image = RF(image, tf.ones_like(mask), name='k_image')

		m_real = mask[:,0:1,...]
		m_imag = mask[:,0:1,...]
		m_mask = tf.concat([m_real, m_imag], axis=1)
		print mask, k_recon, k_image
		condition = tf.cast(tf.real(m_mask)>0.9, tf.bool)
		# where(
		#     condition,
		#     x=None,
		#     y=None,
		#     name=None
		# )
		#Return the elements, either from x or y, depending on the condition.
		k_return  = tf.where(condition, k_image, k_recon, name='k_return')
		updated = FhRh(k_return, tf.ones_like(mask), name=name)
	return tf.identity(updated, name=name)


###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv0', chan, padding='SAME')
				# .Dropout('drop', 0.5)
				.Conv2D('conv1', chan/2, padding='SAME')
				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
				# .Dropout('drop', 0.5)
				# .InstanceNorm('inorm')
				()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=1, stride=1):
	with argscope([Conv2D], stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.9)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_enc', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			# .InstanceNorm('inorm')
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Deconv2D('deconv_i', chan, stride=1) 
			.residual('res_dec', chan, first=True)
			.Deconv2D('deconv_o', chan, stride=2) 
			# .InstanceNorm('inorm')
			# .Dropout('drop', 0.9)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img):
	assert img is not None
	# img = tf_complex(img)
	with argscope([Conv2D, Deconv2D], nl=BNLReLU, kernel_shape=4, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		# e0 = Dropout('dr', e0, 0.9)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.9)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', 2, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		l  = (dd)
		return l

###############################################################################
# @auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	# img = tf_complex(img)
	with argscope([Conv2D, Deconv2D], nl=BNLReLU, kernel_shape=4, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=tf.nn.leaky_relu)
		# img = Dropout('dr', img, 0.9)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret


###############################################################################
class ClipCallback(Callback):
	def _setup_graph(self):
		vars = tf.trainable_variables()
		ops = []
		for v in vars:
			n = v.op.name
			if not n.startswith('discrim/'):
				continue
			logger.info("Clip {}".format(n))
			ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
		self._op = tf.group(*ops, name='clip')

	def _trigger_step(self):
		self._op.run()
###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, maskDir, labelDir, size, ratio = 0.1, dtype='float32', is_training=False):
		"""
		Args:
			shapes (list): a list of lists/tuples. Shapes of each component.
			size (int): size of this DataFlow.
			random (bool): whether to randomly generate data every iteration.
				Note that merely generating the data could sometimes be time-consuming!
			dtype (str): data type.
		"""
		# super(FakeData, self).__init__()

		self.dtype    = dtype
		self.imageDir = imageDir
		self.maskDir  = maskDir
		self.labelDir = labelDir
		self.ratio    = ratio
		self._size    = size
		self.is_training = is_training
	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)   
		print self.is_training


	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_reverse = np.random.randint(1,3)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]

		return reverse

	def random_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = np.random.randint(-90,90)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		rotated = rotate(image, random_rotatedeg, axes=(1,2), reshape=False)
		image = rotated
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
		
	def random_crop(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		limit = np.random.randint(1, 12) # Crop pixel
		randy = np.random.randint(0, limit)
		randx = np.random.randint(0, limit)
		cropped = image[:, randy:-(limit-randy), randx:-(limit-randx)]
		return cropped	
	##################################################################
	def get_data(self, shuffle=True):
		# self.reset_state()
		images = glob.glob(self.imageDir + '/*.*')
		# print "images: ", images
		if self.maskDir:
			masks  = glob.glob(self.maskDir + '/*.*')
		# print "masks: ", masks
		labels = glob.glob(self.labelDir + '/*.*')
		# print "labels: ", labels
		from natsort import natsorted
		images = natsorted(images)
		if self.maskDir:
			masks  = natsorted(masks)
		labels = natsorted(labels)
		# print images
		# print labels

		for k in range(self._size):
			if self.is_training:
				from random import randrange
				rand_index_image = randrange(0, len(images))
				if self.maskDir:
					rand_index_mask  = randrange(0, len(masks))
				rand_index_label = randrange(0, len(labels))
				# rand_index = randrange(0, len(images))
			else:
				rand_index_image = k
				rand_index_mask  = 0
				rand_index_label = k

			image = skimage.io.imread(images[rand_index_image])
			if self.maskDir:
				mask  = skimage.io.imread(masks[rand_index_mask])
			else:
				mask = 255*self.generateMask(DIMZ, DIMY, DIMX, sampling_rate=self.ratio)
			label = skimage.io.imread(labels[rand_index_label])
			
			# print images[rand_index_image], masks[rand_index_mask], labels[rand_index_label]
			# print image.shape, mask.shape, label.shape

			# # Process the static image, make 2 channel image identical
			if image.ndim == 2:
				image = np.stack((image, np.zeros_like(image)), axis=0)
			if mask.ndim == 2:
				mask = np.stack((mask, np.zeros_like(mask)), axis=0)
			if label.ndim == 2:
				label = np.stack((label, np.zeros_like(label)), axis=0)



			seed_image = np.random.randint(0, 2015)
			seed_mask  = np.random.randint(0, 2015)
			seed_label = np.random.randint(0, 2015)

			if self.is_training:
				# pass
				#TODO: augmentation here	

				image = self.random_square_rotate(image, seed=seed_image)
				image = self.random_flip(image, seed=seed_image)
				image = self.random_crop(image, seed=seed_image)


				label = self.random_square_rotate(label, seed=seed_label)
				label = self.random_flip(label, seed=seed_label)
				label = self.random_crop(label, seed=seed_label)


			image = skimage.transform.resize(image, output_shape=(DIMZ, DIMY, DIMX), 
											 order=1, preserve_range=True)
			label = skimage.transform.resize(label, output_shape=(DIMZ, DIMY, DIMX), 
											 order=1, preserve_range=True)

			image = np.expand_dims(image, axis=0)
			mask  = np.expand_dims(mask, axis=0)
			label = np.expand_dims(label, axis=0)



			# yield [image.astype(np.complex64), mask.astype(np.complex64), label.astype(np.complex64)]
			yield [image.astype(np.uint8), 
				   mask.astype(np.uint8), 
				   label.astype(np.uint8)]


def get_data(imageDir, maskDir, labelDir, size=EPOCH_SIZE):
	ds_train = ImageDataFlow(imageDir, 
							 maskDir, 
							 labelDir, 
							 size, 
							 ratio=0.1,
							 is_training=True
							 )


	ds_valid = ImageDataFlow(imageDir.replace('train', 'valid'), 
							 maskDir, 
							 labelDir.replace('train', 'valid'), 
							 size, 
							 ratio=0.1,
							 is_training=False
							 )

	return ds_train, ds_valid

	