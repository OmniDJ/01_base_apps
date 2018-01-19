# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:28:04 2017

"""

import os
#import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
#from PIL import Image

import tensorflow as tf

import numpy as np

from keras.applications import vgg19  # for neural transfer styling
from keras.applications import inception_v3  # for deep dreaming  styling
import keras.backend as K

from scipy.optimize import fmin_l_bfgs_b


from skimage import io


__version__ = "0.1.tf14"
__author__  = "Andrei Ionut Damian"
__copyright__ = "(C) Knowledge Investment Group"
__project__ = "OmniDJ"

#
#TODO: try only with 4,5 layers !!!
#

KERAS_VGG19_STYLE = [
      ('block1_conv1', 1.0), 
      ('block2_conv1', 1.0),
      ('block3_conv1', 1.0), 
      ('block4_conv1', 1.0),
      ('block5_conv1', 1.0),    
    ]
KERAS_VGG19_CONTENT = 'block5_conv2'

TF_VGG19_STYLE = STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
TF_VGG19_CONTENT = 'conv4_2'
TF_MODEL = 'imagenet-vgg-verydeep-19.mat'



INCEPTION_V3_STYLING_TEMPLATE = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
    
    
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
  logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
  print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib


class CONFIG:
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 



  
    
class NeuralStyler:
  """
  Neural Styler based on several methods:
  constructor: 
    method:          
      'tf': uses tf adam optimization
      'keras': uses bfgs scypy optimization
  
  generate:
    np_source: numpy source image
    np_style : numpy style image
    returns generated numpy image (HWC) based on Neural Style Transfer approach
    
  generate_dream:
    np_source: numpy source image
    returns generated numpy image (HWC) based on Deep Dream approach
  
  
  """
  def __init__(self, content_weight = 0.025, style_weight = 1.0, 
               total_var_weight = 1.0,
               method = 'keras',
               config_file = 'config.txt',
               style_layer_weights = None, 
               epochs = 1, nr_iters = 20,
               clear_session = True, 
               DEBUG = True):
    
    self.DEBUG = DEBUG

    self.logger = load_module('logger','logger.py').Logger(
        lib_name = "NEUS", config_file = config_file, DEBUG = self.DEBUG )

    self.method = method
    if self.method == 'tf':   
      tf.reset_default_graph()
      self.sess = tf.Session()
      self._style_layers = TF_VGG19_STYLE
      self._content_layer = TF_VGG19_CONTENT
      self._tf_model_file = self.logger.config_data['TF_MODEL']
    elif self.method == 'keras':
      K.clear_session()
      K.set_image_data_format('channels_last')
      self.sess = K.get_session()
      self._style_layers = KERAS_VGG19_STYLE
      self._content_layer = KERAS_VGG19_CONTENT
    else:
      raise Exception("Unknown method '{}'".format(method))
    
    if style_layer_weights is not None:
      for i,w in enumerate(style_layer_weights):
        self._style_layers[i][1] = w             
    
    self.epochs = epochs
    self.nr_iters = nr_iters
    self.content_weight = content_weight
    self.style_weight = style_weight
    self.total_var_weight = total_var_weight
    self.img_ncols = 400
    self.img_nrows = int(1.618 * self.img_ncols)
    return
  
  def log(self, s, show_time = False):
    self.logger.VerboseLog(s, show_time=show_time)
    return
  
  def generate(self, np_source, np_style):
    """
     Main "fit-transform" function
    """
    assert len(np_source.shape) == 3
    assert len(np_style.shape) == 3
    self.img_ncols = np_source.shape[0]
    self.img_nrows = np_source.shape[1]
    if self.method == 'tf':
      np_result = self.tf_generate(np_source, np_style)
    elif self.method == 'keras':
      np_result = self.k_generate(np_source, np_style)
    return np_result
    
  ###
  ### 
  ### Keras version
  ###    
  ###
  
  def k_generate(self,np_source, np_style):
    self.logger.VerboseLog("Preparing Generic(Keras) NeuralStyler generator")
    base_image = K.variable(self._k_preprocess_image(np_source))
    style_reference_image = K.variable(self._k_preprocess_image(np_style))    
    if K.image_data_format() == 'channels_first':
      combination_image = K.placeholder((1, 3, self.img_nrows, self.img_ncols))
    else:
      combination_image = K.placeholder((1, self.img_nrows, self.img_ncols, 3))    
    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)

    # feed input_tensor to VGG model - just like feeding a batch of 3 images
    self.logger.VerboseLog("Loading VGG19 CNN pretrained on imagenet...")
    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)    
    self.logger.VerboseLog("Done loading VGG19 CNN pretrained on imagenet...", show_time = True)
    
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    loss = K.variable(0.)
    layer_features = outputs_dict[self._content_layer]
    # now layer_features contains results of the three input images for 
    # proposed content feature map (block5_conv2)

    base_image_features = layer_features[0, :, :, :] # get content (1st in batch)
    combination_features = layer_features[2, :, :, :] # fet generated (3rd in batch)
    loss += self.content_weight * self._k_compute_content_cost(base_image_features,
                                                              combination_features)
    
    for layer_name,layer_weight in self._style_layers:
      layer_features = outputs_dict[layer_name]
      style_reference_features = layer_features[1, :, :, :]
      combination_features = layer_features[2, :, :, :]
      sl = self._k_compute_layer_style_cost(style_reference_features, 
                                           combination_features)
      loss += (self.style_weight / len(self._style_layers)) * sl * layer_weight
      
    loss += self.total_var_weight * self._k_total_variation_loss(combination_image)
    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)
    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    # now prepare a function that will evaluate outputs (gradients) 
    # based on combination image placeholder (in tf call Session.run)
    self.f_get_outputs = K.function([combination_image], outputs)
    self._k_loss_value = None
    self._k_grad_value = None
    self.logger.VerboseLog("Done preparing Generic(Keras) NeuralStyler generator.")    

    self.logger.VerboseLog("Running generation process...")    
    self.logger.start_timer('KERAS_GEN')
    x = self._k_preprocess_image(np_source)  # generated image starts from source
    for epoch in range(self.epochs):
      self.logger.VerboseLog(" Epoch {}. Running L-BFGS step...".format(epoch))
      x, min_val, info = fmin_l_bfgs_b(self._k_loss, 
                                       x.flatten(),
                                       fprime = self._k_grads, 
                                       maxfun = self.nr_iters)      
      self.logger.VerboseLog(" Done running L-BFGS step. Loss: {} ".format(
                                    min_val, info['warnflag'],info['task']), show_time = True)      
      self.logger.VerboseLog("   Warn/Task: {}={}".format(info['warnflag'],info['task'][-30:]))
      img = self._k_deprocess_image(x.copy())
      self.logger.OutputImage(img, label = 'keras')
      if self.DEBUG:
        plt.imshow(img) 
    tmr = self.logger.end_timer('KERAS_GEN')
    self.logger.VerboseLog("Done running Keras generation process. Time: {:.2f}s".format(tmr))        
    return img
  
  def _k_loss(self, x):
    assert self._k_loss_value is None
    self._k_loss_value, self._k_grad_value = self._k_eval_loss_and_grads(x)
    return self._k_loss_value
  
  def _k_grads(self, x):
    assert self._k_loss_value is not None
    grad_value = np.copy(self._k_grad_value)
    self._k_loss_value = None
    self._k_grad_value = None
    return grad_value
    
  
  def _k_eval_loss_and_grads(self, x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
    else:
        x = x.reshape((1, self.img_nrows, self.img_ncols, 3))
    outs = self.f_get_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values
  
  def _k_gram_matrix(self, x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
    
    
  def _k_compute_content_cost(self, t_C, t_G):
    return K.sum(K.square(t_C - t_G))

  
  def _k_compute_layer_style_cost(self, t_S, t_G):
    assert K.ndim(t_S) == 3
    assert K.ndim(t_G) == 3
    S = self._k_gram_matrix(t_S)
    C = self._k_gram_matrix(t_G)
    channels = 3
    size = self.img_nrows * self.img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

  def _k_total_variation_loss(self, x):
    assert K.ndim(x) == 4
    rows = self.img_nrows
    cols = self.img_ncols
    if K.image_data_format() == 'channels_first':
      a = K.square(x[:, :, :rows - 1, :cols - 1] - x[:, :, 1:, :cols - 1])
      b = K.square(x[:, :, :rows - 1, :cols - 1] - x[:, :, :rows - 1, 1:])
    else:
      a = K.square(x[:, :rows - 1, :cols - 1, :] - x[:, 1:, :cols - 1, :])
      b = K.square(x[:, :rows - 1, :cols - 1, :] - x[:, :rows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
  
  
  def _k_preprocess_image(self, np_img):
    img = scipy.misc.imresize(np_img, size = (self.img_nrows, self.img_nrows))
    img = np.expand_dims(img, axis=0).astype(float)
    img = vgg19.preprocess_input(img)
    return img  
  
  def _k_deprocess_image(self, x):
    if K.image_data_format() == 'channels_first':
      x = x.reshape((3, self.img_nrows, self.img_ncols))
      x = x.transpose((1, 2, 0))
    else:
      x = x.reshape((self.img_nrows, self.img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
  
  
  ###
  ###
  ### Tensorflow version
  ###
  ###

  def _tf_load_vgg_model(self, path):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    
    vgg = scipy.io.loadmat(path)
  
    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b
  
        return W, b
  
    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)
  
    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
  
    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))
  
    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, self.img_nrows, self.img_ncols, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph



  def _tf_deprocess_image(self, image):
    
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS    
    # Clip image
    image = np.clip(image[0], 0, 255).astype('uint8')
    return image
  
  def _tf_generate_noise_image(self,content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, self.img_nrows, self.img_ncols, CONFIG.COLOR_CHANNELS)).astype('float32')
    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image
  
  
  def _tf_reshape_and_normalize_image(self,image):
    """
    Reshape and normalize the input image (content or style)
    """
    assert len(image.shape) == 3
    if (image.shape[0] != self.img_nrows) or (image.shape[1] != self.img_ncols):
        image = scipy.misc.imresize(image, size = (self.img_nrows, self.img_ncols))
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    
    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS
    
    return image
  
  def _tf_compute_content_cost(self, t_C, t_G):
    """
     computes content cost based on two tensors (content and generated)
     rrturns scalar tensor
     
     actually a_C will be a numpy array that will be transformed into a tensor
    """
    m, n_H, n_W, n_C = t_G.get_shape().as_list()    
    t_C_unrolled = tf.reshape(tf.transpose(t_C), (n_C, n_H*n_W))
    t_G_unrolled = tf.reshape(tf.transpose(t_G), (n_C, n_H*n_W))    
    t_layer_content_loss = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(t_C_unrolled, t_G_unrolled)))    
    return t_layer_content_loss    
  
    
  def _tf_compute_gram(self, t_A):
    """
    computes gram matrix tensor of t_A tensor (matrix)
    """
    t_GA = tf.matmul(t_A, tf.transpose(t_A))
    return t_GA
    
  
  def _tf_compute_layer_style_cost(self, t_S, t_G):
    """
     computes style loss given Style and Generated tensors
     again t_S will be a numpy
    """
    m, n_H, n_W, n_C = t_G.get_shape().as_list()
    
    t_S_unrolled = tf.reshape(tf.transpose(t_S), (n_C, n_H * n_W))
    t_G_unrolled = tf.reshape(tf.transpose(t_G), (n_C, n_H * n_W))

    t_GS = self._tf_compute_gram(t_S_unrolled)
    t_GG = self._tf_compute_gram(t_G_unrolled)

    t_layer_style_loss = (1 / (4 * n_C**2 * (n_H * n_W)**2)) * (
        tf.reduce_sum(tf.square(tf.subtract(t_GS, t_GG))) )
        
    return t_layer_style_loss    
  
  def _tf_compute_style_cost(self, layers_dict, style_touples):
    """
    The style of an image can be represented using the Gram matrix of a hidden 
    layer's activations. However, we get even better results combining this 
    representation from multiple different layers. 
    This is in contrast to the content representation, where usually using just 
    a single hidden layer is sufficient.
    """
    J_style = 0
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in style_touples:

        # Select the output tensor of the currently selected layer
        out = layers_dict[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = self.sess.run(out)
        #print(a_S)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = self._tf_compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style    
  
  def _tf_total_cost(self,J_content, J_style, alpha = 10, beta = 40):
    """
    compute total overall cost
    """
    J = alpha * J_content + beta * J_style
    return J
  
  def tf_generate(self,np_content, np_style):
    """
    generates the output neural styler based on np_content (None, H, W, 3) and
    np_style (None, H, W, 3)
    """
    self.logger.VerboseLog('Preparing TensorFlow model for NeuralStyler...')
    model_file = self.logger.GetDataFile(self._tf_model_file)
    assert model_file is not None
    
    np_content = self._tf_reshape_and_normalize_image(np_content)
    np_style = self._tf_reshape_and_normalize_image(np_style)
    np_generated_source = self._tf_generate_noise_image(np_content)
    
    self.logger.VerboseLog(" Loading [...{}]...".format(model_file[-40:]))
    model = self._tf_load_vgg_model(model_file)
    self.logger.VerboseLog(" Done loading.".format(model_file), show_time = True)
    
    self.logger.VerboseLog("Prep content ...")
    # now prepare the content
    self.sess.run(model['input'].assign(np_content))
    out = model[self._content_layer]
    np_C = self.sess.run(out) 
    # done preparing content
    self.logger.VerboseLog("Done prep content.", show_time = True)
    
    # preparing generated image output tensor
    tf_G = model[self._content_layer]
    # done generated image output tensor
    
    J_content = self._tf_compute_content_cost(np_C, tf_G)
    
    self.logger.VerboseLog("Prep style...")
    # now prepare style tensor
    self.sess.run(model['input'].assign(np_style))
    J_style = self._tf_compute_style_cost(model, self._style_layers)
    # done style tensor
    self.logger.VerboseLog("Done prep style.", show_time = True)
    
    # now finally prepare total cost function
    J = self._tf_total_cost(J_content, J_style, self.content_weight, self.style_weight)
    #       

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    self.logger.VerboseLog('Done preparing TensorFlow model for NeuralStyler.')
    
    self.logger.VerboseLog('Optimizing TensorFlow model for NeuralStyler...')
    self.logger.start_timer("TF_OPTIMIZE")
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(model['input'].assign(np_generated_source))    
    for epoch in range(self.epochs):
      self.logger.start_timer("TF_EPOCH")
      for i in range(self.nr_iters):
        _, loss, loss_style, loss_content = self.sess.run([train_step, 
                                                           J, 
                                                           J_style, 
                                                           J_content])
      tmr1 = self.logger.end_timer("TF_EPOCH")
      self.logger.VerboseLog("Epoch {} [{:.2f}s]: Loss:{} ContentL:{} StyleL:{}".format(
          epoch, tmr1, loss, loss_style, loss_content))
      np_generated = self.sess.run(model['input'])
      img = self._tf_deprocess_image(np_generated)
      self.logger.OutputImage(img, label = 'tf')
      if self.DEBUG:
        plt.imshow(img) 
      
    tmr2 = self.logger.end_timer("TF_OPTIMIZE")
    self.logger.VerboseLog('Done optimizing TensorFlow model for NeuralStyler in {:.2f}s'.format(
        tmr2))
    
    return img
  
  ###
  ### deep dream styler section
  ###
  
  def _k_dd_preprocess_image(self, np_img):
    """
    """
    np_img = np.expand_dims(np_img, axis = 0).astype(float)
    np_img = inception_v3.preprocess_input(np_img)
    return np_img
  
  def _k_dd_deprocess_image(self, np_img):
    """
    Util function to convert a tensor into a valid image.
    """
    if K.image_data_format() == 'channels_first':
        np_img = np_img.reshape((3, np_img.shape[2], np_img.shape[3]))
        np_img = np_img.transpose((1, 2, 0))
    else:
        np_img = np_img.reshape((np_img.shape[1], np_img.shape[2], 3))
    np_img /= 2.
    np_img += 0.5
    np_img *= 255.
    np_img = np.clip(np_img, 0, 255).astype('uint8')
    return np_img   


  def _k_dd_eval_loss_and_grads(self, x):
    """
    """
    outs = self._k_dd_fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values    

  def _k_dd_resize_img(self, img, size):
    """
    """
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)
  
  
  def _k_dd_gradient_ascent(self, x, iterations, step, max_loss=None):
    for i in range(iterations):
      loss_value, grad_values = self._k_dd_eval_loss_and_grads(x)
      if max_loss is not None and loss_value > max_loss:
          break
      if (i % 5) == 0:
        self.log('  DD optimizer Loss value at step {} is {:.2f}'.format(i, loss_value))
      x += step * grad_values
    return x  

  
  def _k_dd_generate(self, np_img, 
                     # Playing with the following hyperparameters will achieve new effects
                     style_template = INCEPTION_V3_STYLING_TEMPLATE,
                     step = 0.01,  # Gradient ascent step size
                     num_octave = 3,  # Number of scales at which to run gradient ascent
                     octave_scale = 1.4,  # Size ratio between scales
                     iterations = 20,  # Number of ascent steps per scale
                     max_loss = 10.,
                     ):
    K.set_learning_phase(0)
    self.log("Loading InceptionV3...")
    model = inception_v3.InceptionV3(weights='imagenet',
                                     include_top=False)
    dream = model.input
    self.log("Done loading InceptionV3", show_time = True)
    self.log("Deep Dream keras graph preparation...")
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    loss = K.variable(0.)
    
    for layer_name, coeff in style_template['features'].items():
      # Add the L2 norm of the features of a layer to the loss.
      assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
      x = layer_dict[layer_name].output
      # We avoid border artifacts by only involving non-border pixels in the loss.
      scaling = K.prod(K.cast(K.shape(x), 'float32'))
      if K.image_data_format() == 'channels_first':
          loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
      else:
          loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
          
    # now compute the gradients of the above loss function with respect to the dream
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())    
    
    outputs = [loss, grads]
    self._k_dd_fetch_loss_and_grads = K.function([dream], outputs)
    self.log("Done Deep Dream keras graph preparation.", show_time = True)


      
    img = self._k_dd_preprocess_image(np_img)
    if K.image_data_format() == 'channels_first':
      original_shape = img.shape[2:]
    else:
      original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    
    for i in range(1, num_octave):
      shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
      successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = self._k_dd_resize_img(img, successive_shapes[0])

    self.log("Processing image for dd styling..")
    self.logger.start_timer('DDOptimizer')
    for shape in successive_shapes:
      self.log(' DD Processing image shape {}'.format(shape))
      self.logger.start_timer(" DDGradStep")
      img = self._k_dd_resize_img(img, shape)
      img = self._k_dd_gradient_ascent(img,
                                       iterations = iterations,
                                       step=step,
                                       max_loss=max_loss)
      upscaled_shrunk_original_img = self._k_dd_resize_img(shrunk_original_img, shape)
      same_size_original = self._k_dd_resize_img(original_img, shape)
      lost_detail = same_size_original - upscaled_shrunk_original_img
    
      img += lost_detail
      shrunk_original_img = self._k_dd_resize_img(original_img, shape)    
      tmr = self.logger.end_timer(" DDGradStep")
      self.log(' DD Processing image step finished in {:.2f}s'.format(tmr))
    tmra = self.logger.end_timer('DDOptimizer')
    self.log("Done processing image for dd styling in {:.2f}s".format(tmra))
    
    final_img = self._k_dd_deprocess_image(np.copy(img))
    self.logger.OutputImage(final_img, label = 'keras_dd')    
    return final_img
  
  def generate_dream(self, np_img):
    """
    Generates a deep-dream-like neural styled image based on np_image
    """
    assert self.method == 'keras', 'Only keras supported for deep dream generation'
    np_res = self._k_dd_generate(np_img)
    return np_res
      
    
  ###
  ###
  ###
  

  
  
if __name__ == '__main__':
  tests = ['DREAM','TRANSFER']
  test = 0
  np_cont = io.imread('cont.png')
  np_style = io.imread('style.png')
  
  if tests[test] == 'TRANSFER':
    methods = ['tf', 'keras']  
    for method in methods:
      if method == 'keras':
        content_weight = 0.025
        style_weight = 1.5
      else:
        content_weight = 10
        style_weight = 60    
      neus_gen = NeuralStyler(epochs = 1,
                              method = method, 
                              content_weight = content_weight,
                              style_weight = style_weight)
      np_output = neus_gen.generate(np_cont, np_style)
      plt.imshow(np_output)  
      plt.show()
  else:
    neus_gen = NeuralStyler(epochs = 1,
                            method = 'keras')    
    np_output = neus_gen.generate_dream(np_cont)
    plt.imshow(np_output)  
    plt.show()
    
