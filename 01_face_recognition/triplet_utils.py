#
#
# Triple loss helper functions
#
#

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge


def identity_loss(y_true, y_pred):
  """
  return just the prediction
  model must output the actual loss
  usefull for models where the trained model is not the actual 
  inference model
  """
  return K.mean(y_pred - 0 * y_true)
  

def bpr_triplet_loss(X):
  """
  computes BPR triple loss. this loss function will NOT be used in 
  model.compile
  returns scalar valued tensor
  
  [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking 
  from implicit feedback." Proceedings of the Twenty-Fifth Conference on 
  Uncertainty in Artificial Intelligence. AUAI Press, 2009.  
  """
  positive_item_latent, negative_item_latent, user_latent = X

  # BPR loss
  loss = 1.0 - K.sigmoid(
      K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
      K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
  return loss


def fr_triplet_loss(X, alpha = 0.2):
  """
  computes triple loss. this loss function will NOT be used in model.compile
  returns scalar valued tensor
  
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))  
  """
  anchor, positive, negative = X
  dist_pos = K.sum(K.square(anchor - positive), axis = -1)
  dist_neg = K.sum(K.square(anchor - negative), axis = -1)
  basic_loss = dist_pos - dist_neg + alpha
  loss = K.sum(K.maximum(0,basic_loss))
  return loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
  """
  computes triplet loss. this function will actually be used in model.compile
  the model must use 3x inference graph
  returns scalar valued tensor
  """
  loss = fr_triplet_loss(y_pred)
  return loss


def build_model(embed_net, obs_shape):
  """
  builds triplet loss trainable model
  must receive input shape and actual embedding calculation network
  embed_net MUST NOT include Input
  """
  anchor_input = Input((obs_shape))
  positive_input = Input((obs_shape))
  negative_input = Input((obs_shape))
  
  embed_anchor = embed_net(anchor_input)
  embed_positive = embed_net(positive_input)
  embed_negative = embed_net(negative_input)
  
  final_layer = merge([embed_anchor, embed_positive, embed_negative])
  model = Model(inputs = [anchor_input, positive_input, negative_input],
                outputs = final_layer)
  model.compile(loss=triplet_loss, optimizer = "adam")
  return model

def build_model_v1(embed_net, obs_shape):
  """
  builds model that outputs triplet loss
  and uses identity loss to "fool" the optimizer into using the network output
  as the cost function
  must receive input shape and actual embedding calculation network
  embed_net MUST NOT include Input
  """
  anchor_input = Input((obs_shape))
  positive_input = Input((obs_shape))
  negative_input = Input((obs_shape))
  
  embed_anchor = embed_net(anchor_input)
  embed_positive = embed_net(positive_input)
  embed_negative = embed_net(negative_input)
  
  final_layer = merge([embed_anchor, embed_positive, embed_negative],
                      mode = bpr_triplet_loss)
  model = Model(inputs = [anchor_input, positive_input, negative_input],
                outputs = final_layer)
  model.compile(loss=identity_loss, optimizer = "adam")
  return model

def train_fr_model(embed_net, X_data, triplet_generator, epochs):
  """
  trains embedding network by using:
    triplet model builder 
    data and generator function that must return the exact number and shape of observations
    that model input is expecting
    X_data.shape must be (None, H, W, C)
  """
  H = X_data.shape[1]
  W = X_data.shape[2]
  C = X_data.shape[3]
  
  model = build_model(embed_net, (H,W,C))
  for epoch in range(epochs):
    X_batch = triplet_generator(X_data)
    model.fit(X_batch, epochs = 1)
  return embed_net
  
  
