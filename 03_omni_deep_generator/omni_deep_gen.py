# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:15:18 2017

@author: Andrei
"""
import tensorflow as tf

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

from keras.datasets import mnist, cifar100
from sklearn.datasets import fetch_lfw_people


import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from odg_utils import LoadLogger

__module__  = "DeepGenerator" 
__description__ = "OmniDeepGenerator engine for generating images"
__version__ = "0.0.1"
__author__  = "Andrei Ionut Damian"
__copyright__ = "(C) Knowledge Investment Group"
__project__ = "OmniDJ"




class ImageDataGenerator:
  """
    Data generator class for Omni Deep Generator engine
  """
  def __init__(self, data_folder, label_file):
    self.data_folder = data_folder
    self.label_file = label_file
    
    return
  
  def get_batches(self):
    """
     yelds next batch in form (X_batch, y_batch)
    """
    for batch in batches:
      prepared_batch = (X_batch, y_batch)
      yield prepared_batch
    return

class DeepVAE:
  """
    OmniDeepGenerator based on Variational Auto Encoders
  """
  def __init__(self, latent_space_dim = 2,
               name = 'vae'):
    
    K.clear_session()
    
    self.prepared = False
    self.trained = False
    self.__version__ = __version__
    self.__app__ = "ODGVAE"
    self.ls_dim = latent_space_dim  # size of encoding
    self.logger = LoadLogger(lib_name = self.__app__)
    self.name = name
    

    self.tf_decoder_output = None
    self.tf_decoder_input = None
    
    self.tf_encoder_input = None
    self.tf_encoder_output = None
    self.tf_encoder_mean = None
    self.tf_encoder_var = None

    self.tf_vae_graph = None
    self.tf_vae_output = None
    self.session = None

    self.encoder_input_name = 'encoder_input'
    self.encoder_out_mean_name = 'encoder_out_mean'
    self.encoder_out_var_name = 'encoder_out_var'
    self.encoder_output_name = 'encoder_z'

    self.decoder_input_name ='decoder_input'
    self.decoder_output_name = 'decoder_output'
    
    self.input_shape_name = 'image_shape'
    
    self.config_data = {}
    self.config_data[self.encoder_input_name] = self.encoder_input_name
    self.config_data[self.encoder_out_mean_name] = self.encoder_out_mean_name
    self.config_data[self.encoder_out_var_name] = self.encoder_out_var_name
    self.config_data[self.encoder_output_name] = self.encoder_output_name
    self.config_data[self.decoder_input_name] = self.decoder_input_name
    self.config_data[self.decoder_output_name] = self.decoder_output_name
    
    self.load_model()
    return
  

  def _identity_loss(self, y_true, y_pred):
    """
    return just the prediction
    model must output the actual loss
    usefull for models where the trained model is not the actual 
    inference model
    """
    return K.mean(y_pred - 0 * y_true)
  
  
  def _sampling_layer_func(self, args):
    """
    
    """
    z_mean, z_log_var = args
    # now generate a gaussian
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],  # batch dim
                                     self.ls_dim),        # layer size
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


  def _vae_loss(self, x_input, z_decoded):
    x_input = K.flatten(x_input)
    z_decoded = K.flatten(z_decoded)
    xent_loss = keras.metrics.binary_crossentropy(x_input, z_decoded)
    kl_loss = -5e-4 * K.mean(
        1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
    loss = K.mean(xent_loss + kl_loss)    
    return loss
  
  def _vae_loss_merge(self, x_merged):
    x_input, z_decoded = x_merged
    x_input = K.reshape(x_input, (-1, self.nr_preds ))
    z_decoded = K.reshape(z_decoded, (-1, self.nr_preds ))
    xent_loss = K.mean(K.binary_crossentropy(x_input, z_decoded), axis=1,
                       keepdims = True)
    kl_loss = -5e-4 * K.mean(
        1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1, 
        keepdims = True)
    losses = xent_loss + kl_loss #K.mean(xent_loss + kl_loss)    
    return losses
  
  
  def preprocess_images(self, np_images):
    x = np_images.astype(np.float32) / 255.
    return x
  
  def deprocess_images(self, np_images):
    x = np.clip(np_images * 255, 0, 255).astype('uint8')
    return x
  
  def _prepare_model(self, input_shape, force_prepare = False):
    """
    """
    if self.trained and (not force_prepare):
      self.log("Model allready trained. Skipping preparation...")
      return
    
    assert len(input_shape) == 3, "Input shape must be HWC"
    nr_channels = input_shape[2]
    self.config_data[self.input_shape_name] = input_shape
    self.input_shape = input_shape
    self.nr_preds = np.prod(input_shape)
    #
    # encoder part
    #
    
    input_layer = layers.Input(input_shape, name = self.config_data[self.encoder_input_name] )  
    self.config_data[self.encoder_input_name] = input_layer.name
    
    x = layers.Conv2D(32, 3,
                      padding='same', activation='relu')(input_layer)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu',
                      strides=(2, 2))(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    
    self.z_mean = layers.Dense(self.ls_dim, name = self.config_data[self.encoder_out_mean_name])(x)
    self.config_data[self.encoder_out_mean_name] = self.z_mean.name
    
    self.z_log_var = layers.Dense(self.ls_dim, name = self.config_data[self.encoder_out_var_name])(x)    
    self.config_data[self.encoder_out_var_name] = self.z_log_var.name
    
    z = layers.Lambda(self._sampling_layer_func, 
                      name = self.config_data[self.encoder_output_name])([self.z_mean, self.z_log_var])
    self.config_data[self.encoder_output_name] = z.name
    
    
    #
    # decoder part
    #
    decoder_input = layers.Input(K.int_shape(z)[1:], name = self.config_data[self.decoder_input_name])
    
    # Upsample to the correct number of units
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)
    
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = layers.Reshape(shape_before_flattening[1:])(x)
    
    # We then apply then reverse operation to the initial
    # stack of convolution layers: a `Conv2DTranspose` layers
    # with corresponding parameters.
    x = layers.Conv2DTranspose(32, 3,
                               padding='same', activation='relu',
                               strides=(2, 2))(x)
    x = layers.Conv2D(nr_channels, 3,
                      padding='same', activation='sigmoid',
                      name = self.config_data[self.decoder_output_name])(x)
    
    # We end up with a feature map of the same size as the original input.
    
    # This is our decoder model.
    self.decoder = Model(decoder_input, x)
    
    # We then apply it to `z` to recover the decoded `z`.
    z_decoded = self.decoder(z)  
    
    y = layers.Lambda(self._vae_loss_merge)([input_layer, z_decoded])
    
    #y = layers.Lambda([input_layer, z_decoded], 
    #                 mode = self._vae_loss_merge,
    #                 output_shape = (None, 1))
    
    self.vae = Model(inputs = input_layer, outputs = y)
    
    self.vae.compile(optimizer = 'rmsprop', loss = self._identity_loss)
    
    self.logger.LogKerasModel(self.vae)
    
    self.config_data[self.decoder_input_name] = decoder_input.name
    self.config_data[self.decoder_output_name] = self.vae.layers[-2].outputs[0].name # x.name
    
    self.session = K.get_session()
    self.tf_decoder_output = x
    self.tf_decoder_input = decoder_input
    
    self.tf_encoder_input = input_layer
    self.tf_encoder_output = z
    self.tf_encoder_mean = self.z_mean
    self.tf_encoder_var = self.z_log_var

    self.tf_vae_graph = self.vae.output.graph
    self.tf_vae_output = self.vae.output
    
    self.prepared = True
    self.show_tensor_names()
    return
  
  
  def log(self, s, show_time = False):
    self.logger.VerboseLog(s, show_time=show_time)
    return
  
  
  def show_tensor_names(self):
    self.log("Full VAE graph tensor names and properties:")
    for key,val in self.config_data.items():
      self.log("  {}:".format(key).ljust(21)+"{}".format(val))
    return
  
  
  def train(self,X_train, X_valid, epochs = 10, batch_size = 16,
            y_train = None, y_test = None, force_train = False,
            preprocess_images = True):
    
    if self.trained and (not force_train):
      self.log("Model allready trained. Skipping fit.")
      return
    
    assert len(X_train.shape)==4, "Input data must be NHWC"
    
    input_shape = X_train.shape[1:]
    
    if preprocess_images:
      X_train = self.preprocess_images(X_train)
      X_valid =self.preprocess_images(X_valid)
    
    if y_train is None:
      y_train = np.zeros((X_train.shape[0],1))
      
    if y_test is None:
      y_test = np.zeros((X_valid.shape[0],1))
    
    if not self.prepared:
      self._prepare_model(input_shape = input_shape)
      
    if self.prepared:
      self.log("Training with batch {} for {} epochs".format(batch_size, epochs))
      self.save_model()
      self.vae.fit(x = X_train,
                   y = y_train,
                   shuffle = True,
                   epochs = epochs,
                   batch_size = batch_size,
                   validation_data = (X_valid, y_test),
                   callbacks = self.logger.GetStandardKerasCallbacks(name = self.name))
      self.trained = True
    return self.vae
  
  
  def train_generator(self, data_generator):
    return
  
  
  def save_model(self, name = None,):
    assert self.prepared
    assert self.trained    
    if name is not None: self.name = name
    tensor_list = [
                   self.tf_vae_output, 
                   self.tf_decoder_output, 
                   self.tf_encoder_output,
                   ]
    self.logger.SaveGraphToModels(session = self.session, 
                                  tensor_list = tensor_list, 
                                  graph_name = self.name, 
                                  input_names = [self.config_data[self.encoder_input_name],
                                                 self.config_data[self.decoder_input_name]])
    self.save_model_config()
    return
  
  
  def save_model_config(self):
    self.logger.SaveConfigDict(self.name, self.config_data)
    return
  
  def load_model_config(self):
    self.config_data = self.logger.LoadConfigDict(self.name)
    return
    
  
  def setup_from_keras_checkpoint(self):
    self.log("Attempting to load from .h5...")
    self.keras_model = self.logger.LoadKerasModel(self.name)
    if self.keras_model is not None:
      # begin setup from keras model
      self.load_model_config()
      self.tf_vae_graph = self.keras_model.layers[-1].output.graph
      self.log("Model loaded from {}.h5 file".format(self.name))
      # end setup from keras model
    return
  
  def load_model(self, name = None):
    if name is not None: self.name = name
    self.logger.VerboseLog("Loading full VAE graph [{}]...".format(self.name))
    self.tf_vae_graph = self.logger.LoadGraphFromModels(self.name)
    
    if self.tf_vae_graph is None:
      # try to load from .h5
      self.setup_from_keras_checkpoint()
      
    if self.tf_vae_graph is not None:
      g = self.tf_vae_graph
      self.trained = True
      self.prepared = True
      self.load_model_config()
      
      self.tf_decoder_input = g.get_tensor_by_name(self.config_data[self.decoder_input_name])
      self.log(" Prepared {}:".format(self.decoder_input_name,).ljust(30)+
               "{}".format(self.tf_decoder_input))
      
      self.tf_decoder_output = g.get_tensor_by_name(self.config_data[self.decoder_output_name])
      self.log(" Prepared {}:".format(self.decoder_output_name,).ljust(30)+
               "{}".format( self.tf_decoder_output))

      self.tf_encoder_input = g.get_tensor_by_name(self.config_data[self.encoder_input_name])
      self.log(" Prepared {}:".format(self.encoder_input_name,).ljust(30)+
               "{}".format( self.tf_encoder_input))

      self.tf_encoder_output = g.get_tensor_by_name(self.config_data[self.encoder_output_name])
      self.log(" Prepared {}:".format(self.encoder_output_name,).ljust(30)+
               "{}".format( self.tf_encoder_output))

      self.tf_encoder_mean = g.get_tensor_by_name(self.config_data[self.encoder_out_mean_name])
      self.log(" Prepared {}:".format(self.encoder_out_mean_name,).ljust(30)+
               "{}".format( self.tf_encoder_mean))

      self.tf_encoder_var = g.get_tensor_by_name(self.config_data[self.encoder_out_var_name])
      self.log(" Prepared {}:".format(self.encoder_out_var_name,).ljust(30)+
               "{}".format( self.tf_encoder_var))
      
      self.input_shape = self.config_data[self.input_shape_name]
      self.log(" InputShape: {}".format(self.input_shape))
      
      self.session = tf.Session(graph = g)
      
      self.logger.VerboseLog("Done loading models and graphs.")
    return    


  def generate_embedding(self, np_image):
    if len(np_image.shape) != 4:
      np_image = np.expand_dims(np_image, axis = 0)
    assert len(np_image.shape) == 4, " np_image must be NHWC or HWC"
    assert self.tf_encoder_mean is not None
    assert self.tf_encoder_var is not None
    emb_mean, emb_var = self.session.run(
                    [self.tf_encoder_mean, self.tf_encoder_var],
                    feed_dict = {
                        self.tf_encoder_input : np_image
                        }
                    )
    emb = emb_mean + np.exp(emb_var)
    return emb, emb_mean, emb_var

  def generate_z_encoding(self, np_image):
    if len(np_image.shape) != 4:
      np_image = np.expand_dims(np_image, axis = 0)
    assert len(np_image.shape) == 4, " np_image must be NHWC or HWC"
    assert self.tf_encoder_output is not None
    embed = self.session.run(self.tf_encoder_output,
                           feed_dict = {
                               self.tf_encoder_input : np_image
                               })
    return embed
    

  def generate_image(self, embed, shape, deprocess = True):
    assert self.tf_decoder_output is not None, "Decoder output tensor is None !"
    #self.log("Generating image ...")
    self.logger.start_timer("Decoder")
    img = self.session.run(self.tf_decoder_output,
                           feed_dict = {
                               self.tf_decoder_input : embed
                               })
    self.logger.start_timer(" DecoderPostProcess")
    img = img.reshape(shape)
    if deprocess:
      img = self.deprocess_images(img)
    self.logger.end_timer(" DecoderPostProcess")
    self.logger.end_timer("Decoder")
    #self.log("Done generating image ...", show_time = True)
    return img
  
  def shutdown(self):
    self.session.close()
    del self.session
    return
  
 
  
  def test_vae(self, test_size = 15, cmap = None):
    if self.trained:
      # Display a 2D manifold of the digits
      n = test_size  # figure with 15x15 digits
      img_size = self.input_shape[0]
      scene = np.zeros((img_size * n, img_size * n, self.input_shape[2]), 
                        dtype = np.uint8)      
      # Linearly spaced coordinates on the unit square were transformed
      # through the inverse CDF (ppf) of the Gaussian
      # to produce values of the latent variables z,
      # since the prior of the latent space is Gaussian
      grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
      grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
      sample_i_j = tuple(np.random.randint(0,n,size=2))
      
      for i, yi in enumerate(grid_x):
          for j, xi in enumerate(grid_y):
              z_sample = np.array([[xi, yi]])
              #z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
              #x_decoded = self.decoder.predict(z_sample, batch_size = batch_size)              
              #digit = x_decoded[0].reshape(digit_size, digit_size)
              img = self.generate_image(embed=z_sample,
                                        shape=self.input_shape)
              scene[i * img_size: (i + 1) * img_size,
                    j * img_size: (j + 1) * img_size] = img
              if (i,j) == sample_i_j:
                sampled_image = img
                sampled_grid = scene[i * img_size: (i + 1) * img_size,
                                     j * img_size: (j + 1) * img_size]
      
      self.logger.show_timers()
      
      plt.figure(figsize=(10, 10))
      scene = np.squeeze(scene) # remove singleton dims on show
      if cmap is None:
        plt.imshow(scene,) 
        plt.show()    
        plt.imshow(sampled_image)
        plt.show()    
        plt.imshow(sampled_grid)
        plt.show()    
      else:
        plt.imshow(scene, cmap=cmap) 
        plt.show()    
        plt.imshow(sampled_image, cmap=cmap) 
        plt.show()    
        plt.imshow(sampled_grid, cmap=cmap) 
        plt.show()            
    return
    
  
class DeepGAN:
  """
    OmniDeepGenerator based on Generative Adversatial Networks
  """
  def __init__(self):
    return
  
  def train(self,np_images, np_labels):
    return
  
  def train_generator(self, data_generator):
    return
  
  
  
if __name__=='__main__':
  tests = ['VAE_MNIST','VAE_CIFAR_PERS', "VAE_FACES"]
  test = tests[2]
  
  print("\n\nRunning {}".format(test), flush = True)
  
  if test == 'VAE_MNIST':
    # Train the VAE on MNIST digits
    batch_size = 128
    epochs = 2
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))    
    ovae = DeepVAE(name = 'mnist_vae_e{}_b{}'.format(epochs, batch_size))
    ovae.train(X_train = x_train, X_valid = x_test, 
                  batch_size = batch_size,
                  epochs = epochs,
                  preprocess_images = True)
    ovae.test_vae(cmap = 'Greys_r')
    if False:
      t = np.random.randint(0, x_test.shape[0]-1)
      print("Testing: {}".format(y_test[t]))
      plt.imshow(x_test[t].reshape((28,28)), cmap = 'gray')
      plt.show()
      emb_dict = {}
      emb_dict['z'] = ovae.generate_z_encoding(x_test[t])
      emb_dict['e'],emb_dict['m'],emb_dict['v'] = ovae.generate_embedding(x_test[t])    
      for key,val in emb_dict.items():
        print("{}: {}".format(key, val))
        img = ovae.generate_image(val, (28,28))
        plt.imshow(img, cmap = 'gray')
        plt.show()
    #ovae.shutdown()
    
  elif test == 'VAE_CIFAR_PERS':
    # Test the VAE on persons within CIFAR100
    batch_size = 16
    epochs = 20
    show_sampling = True
    nr_samples = 3
    
    person_classes = [46, 98] # 14 is "people" in coarse labels, 46: man, 98: woman
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    train_indices = np.isin(y_train, person_classes).ravel()
    test_indices  = np.isin(y_test, person_classes).ravel()
    x_train = x_train[train_indices,:,:,:]
    y_train = y_train[train_indices,:]
    x_test = x_test[test_indices,:,:,:]
    y_test = y_test[test_indices,:]
    if show_sampling:
      x_sample = np.random.choice(np.arange(x_train.shape[0]), size = nr_samples, replace = False)
      for i in range(x_sample.shape[0]):
        plt.imshow(x_train[x_sample[i]])
        plt.show()
      x_sample = np.random.choice(np.arange(x_test.shape[0]), size = nr_samples, replace = False)
      for i in range(x_sample.shape[0]):
        plt.imshow(x_test[x_sample[i]])
        plt.show()
    ovae = DeepVAE(name = 'cifar100_vae_e{}_b{}'.format(epochs, batch_size))
    ovae.train(X_train = x_train, X_valid = x_test, 
               batch_size = batch_size,
               epochs = epochs,
               preprocess_images = True)
    ovae.test_vae()
    #ovae.shutdown()
    
  elif test == 'VAE_FACES':
    batch_size = 64
    epochs = 20
    show_sampling = True
    nr_samples = 3
    
    img_resize = 1
    rect_H = slice(60, 195, None)
    rect_W = slice(60, 195, None)
    img_H = int(len(range(*rect_H.indices(10000))) * img_resize)
    img_W = int(len(range(*rect_W.indices(10000))) * img_resize)
    rect_slice = (rect_H, rect_W)
    print(" LFW size is: {}".format((img_H,img_W)))

    
    ovae = DeepVAE(name = "lfw_vae_{}x{}_e{}_b{}".format(
                                    img_H, img_W, epochs, batch_size))
    
    if not ovae.trained:
      ovae.log("Loading LFW ...")
      lfw_people = fetch_lfw_people(color = True, 
                                    slice_ = rect_slice, 
                                    resize = img_resize)
      ovae.log("Done loading LFW ...", show_time = True)
      x_train, x_test, y_train, y_test = train_test_split(lfw_people.images, 
                                                          lfw_people.target, 
                                                          test_size = 0.1)
      if show_sampling:
        x_sample = np.random.choice(np.arange(x_train.shape[0]), 
                                              size = nr_samples, replace = False)
        for i in range(x_sample.shape[0]):
          plt.imshow(x_train[x_sample[i]].astype(np.uint8))
          plt.show()
        x_sample = np.random.choice(np.arange(x_test.shape[0]), 
                                              size = nr_samples, replace = False)
        for i in range(x_sample.shape[0]):
          plt.imshow(x_test[x_sample[i]].astype(np.uint8))
          plt.show()
      
      ovae.train(X_train = x_train, X_valid = x_test, 
                 batch_size = batch_size,
                 epochs = epochs,
                 preprocess_images = True)
    
    ovae.test_vae()
    #ovae.shutdown()
    
    
    
    
    
    

