# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:08:05 2017

History:
  2017-11-01  First version based on dlib
  2017-12-08  Added TF based inference (using Keras model)
  2017-12-11  Added face alignment

"""
from sklearn.metrics.pairwise import pairwise_distances
from collections import OrderedDict
import cv2
import pandas as pd
import numpy as np
import os

from scipy.misc import imresize

from omni_utils import LoadLogger
from omni_camera_utils import VideoCameraStream, np_rect, np_circle

from omni_utils import FacialLandmarks
from omni_utils import INNER_EYES_AND_BOTTOM_LIP, OUTER_EYES_AND_NOSE
from omni_utils import TEMPLATE, INV_TEMPLATE, TPL_MIN, TPL_MAX, MINMAX_TEMPLATE

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

__module__  = "OmniFR"
__version__ = "0.2.tf14"
__author__  = "Andrei Ionut Damian"
__copyright__ = "(C) Knowledge Investment Group"
__project__ = "OmniDJ"

_METHODS_ = ['dlib', 'tf', 'keras']

try:
  import dlib
except:
  print("Running without dlib")



FACIAL_LANDMARKS = OrderedDict([
	(FacialLandmarks.FL_MOUTH, (48, 68)),
	(FacialLandmarks.FL_REYEB, (17, 22)),
	(FacialLandmarks.FL_LEYEB, (22, 27)),
	(FacialLandmarks.FL_REYE, (36, 42)),
	(FacialLandmarks.FL_LEYE, (42, 48)),
	(FacialLandmarks.FL_NOSE, (27, 35)),
	(FacialLandmarks.FL_JAW, (0, 17))
])
  



def is_shape(name, nr):
  assert name in FacialLandmarks.FL_SET
  return (nr>=FACIAL_LANDMARKS[name][0]) and (nr<FACIAL_LANDMARKS[name][1])

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
  # create two copies of the input image -- one for the
  # overlay and one for the final output image
  overlay = image.copy()
  output = image.copy()
  
  # if the colors list is None, initialize it with a unique
  # color for each facial landmark region
  if colors is None:
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
              (168, 100, 168), (158, 163, 32),
              (163, 38, 32), (180, 42, 220)]
    # loop over the facial landmark regions individually
  for (i, name) in enumerate(FACIAL_LANDMARKS.keys()):
    # grab the (x, y)-coordinates associated with the
    # face landmark
    (j, k) = FACIAL_LANDMARKS[name]
    pts = shape[j:k]
    # check if are supposed to draw the jawline
    if name == "JAW":
      # since the jawline is a non-enclosed facial region,
      # just draw lines between the (x, y)-coordinates
      for l in range(1, len(pts)):
        ptA = tuple(pts[l - 1])
        ptB = tuple(pts[l])
        cv2.line(overlay, ptA, ptB, colors[i], 2)
 
    # otherwise, compute the convex hull of the facial
    # landmark coordinates points and display it
    else:
      hull = cv2.convexHull(pts)
      cv2.drawContours(overlay, [hull], -1, colors[i], -1)    
  # apply the transparent overlay
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
  
  # return the output image
  return output      


class FaceEngine:
  def __init__(self, 
               path_small_shape_model = None, 
               path_large_shape_model = None, 
               path_faceid_model = None,
               path_custom_model = None,
               logger = None,
               fr_method = 'dlib',
               score_threshold = 0.9, 
               DEBUG = False,
               config_file = 'config_omnifr.txt',
               output_file = None):
    """
     loads DLib models for 5, 68 feature detection together with CNN model for 
       128 face embeddings
     loads pretrained tf/keras model (uses tf sess.run for inference)q
     
    """
    
    if not (fr_method in _METHODS_):
      raise Exception("Unknown method: {}".format(fr_method))
    self.method = fr_method
    self.__module__ = __module__
    self.score_threshold = score_threshold
    self.__version__ = __version__
    self.DEBUG = DEBUG
    self._tf_input_HW = (100, 100) # set to a default - used for thumb creation
    self._model_channels = 'channels_last'
    if logger is None:
      self.logger = LoadLogger(self.__module__, 
                               config_file = config_file,
                               DEBUG = self.DEBUG)
    else:
      self.logger = logger
    
    self.log("Initializing FaceEngine v.{}".format(self.__version__))
    self.config_data = self.logger.config_data
    self.shape_large_model = None
    self.shape_small_model = None
    self.faceid_model = None
    self.NR_EMBEDDINGS = 128
    self.ID_FIELD = "ID"
    self.NAME_FIELD = "NAME"
    
    if output_file is None:
      assert "FR_OUTPUT_FILE" in self.config_data.keys()
      output_file = self.config_data["FR_OUTPUT_FILE"]
    self.data_file = os.path.join(self.logger._data_dir, output_file)
    
    self.log("  OmniFR output file: [...{}]".format(self.data_file[-30:]))
    
    self.feats_names = []
    for i in range(self.NR_EMBEDDINGS):
      self.feats_names.append("F_{}".format(i+1))
      
    if os.path.isfile(self.data_file):
      self.log("  Loading faces dataset [...{}]".format(self.data_file[-40:]))
      self.df_faces = pd.read_csv(self.data_file, index_col = False)
      self.columns = list(self.df_faces.columns)
      self.log("  Loaded {} face IDs".format(self.df_faces.shape[0]))
    else:
      self.columns = [self.ID_FIELD, self.NAME_FIELD]
      for i in range(128):
        self.columns.append(self.feats_names[i])        
      self.df_faces = pd.DataFrame(columns = self.columns)
      self.log("  Created empty faces dataset")
    
    if path_large_shape_model is None:
      assert "DLIB_FACE_MODEL" in self.config_data.keys()
      path_large_shape_model = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_MODEL"])      
    self.log("  Loading dlib-68-shape-pred [{}]...".format(path_large_shape_model[-20:]))
    self._shape_large_model = dlib.shape_predictor(path_large_shape_model)
    self.log("  Done loading dlib-68-shape-pred", show_time = True)
      
    if path_small_shape_model is None:
      assert "DLIB_FACE_MODEL_SMALL" in self.config_data.keys()
      path_small_shape_model = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_MODEL_SMALL"])      
    self.log("  Loading dlib-5-shape-pred [{}]...".format(path_small_shape_model[-20:]))
    self._shape_small_model = dlib.shape_predictor(path_small_shape_model)
    self.log("  Done loading dlib-5-shape-pred.", show_time = True)

    self.log("  Loading dlib face detector ...")    
    self._face_detector = dlib.get_frontal_face_detector()
    self.log("  Done loading dlib face detector.", show_time = True)    
    
    
    if self.method == 'dlib':
      if path_faceid_model is None:
        assert "DLIB_FACE_NET" in self.config_data.keys()
        path_faceid_model = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_NET"])
      self.log("  Loading dlib face recognition model [{}]...".format(path_faceid_model[-20:]))
      if hasattr(dlib, "face_recognition_model_v1"):
        self._dlib_face_recog = dlib.face_recognition_model_v1(path_faceid_model)
      else:
        self._dlib_face_recog = None
        ver = 0
        self.log("    Dlib face recognition model NOT available v{}.".format(ver), show_time = True)
      self.log("  Done loading dlib face recognition model.", show_time = True)

    elif self.method == 'keras':    
      if path_custom_model is None:
        assert "KERAS_FACE_NET" in self.config_data.keys() 
        path_custom_model = os.path.join(self.logger._data_dir, self.config_data["KERAS_FACE_NET"])
      self.log("  Loading Keras model [...{}]...".format(path_custom_model[-30:]))
      self._keras_model = load_model(path_custom_model)
      self._tf_fr_output = self._keras_model.output
      self._tf_fr_input = self._keras_model.input
      self.sess = K.get_session()
      self.log("  Done loading Keras model.", show_time = True)
      
      if "KERAS_MODEL_CHANNELS" in self.config_data.keys():
        self._model_channels = self.config_data["KERAS_MODEL_CHANNELS"]
        
      self.log("  Keras model image format: {}".format(self._model_channels))
      
      input_size = int(self.config_data["KERAS_MODEL_INPUT_SIZE"])
      self._tf_input_HW = (input_size, input_size)
      self.log("  Keras model input HW: {}".format(self._tf_input_HW))
    
    elif self.method == 'tf':
      if path_custom_model is None:
        assert "TF_FACE_NET" in self.config_data.keys() 
        path_custom_model = os.path.join(self.logger._modl_dir, self.config_data["TF_FACE_NET"])
      
      self._tf_graph = self.logger.LoadTFGraph(path_custom_model)
      self._tf_tensors_config = self.logger.LoadConfigDict(path_custom_model[:-3])
      
      assert "INPUT_TENSOR" in self._tf_tensors_config.keys()
      assert "OUTPUT_TENSOR" in self._tf_tensors_config.keys()
      assert "LEARNING_PHASE_TENSOR" in self._tf_tensors_config.keys()
      
      self._tf_fr_output = self._tf_graph.get_tensor_by_name(self._tf_tensors_config["OUTPUT_TENSOR"])
      self._tf_fr_input = self._tf_graph.get_tensor_by_name(self._tf_tensors_config["INPUT_TENSOR"])
      self.sess = tf.Session(graph = self._tf_graph)

      # If needed create a config key that specifies TF input images could be [C,H,W]
      self.log("  TF model image format: {}".format(self._model_channels))

      input_size = int(self.config_data["TF_MODEL_INPUT_SIZE"])
      self._tf_input_HW = (input_size, input_size)
      self.log("  TF model input HW: {}".format(self._tf_input_HW))

    else:
      raise Exception("Unknown method for face embedding {}".format(self.method))
    

    self.log("Initialized FaceEngine v.{}".format(self.__version__))
    return


  def log(self,s, show_time = False):
    self.logger.VerboseLog(s, show_time = show_time)
    return
  
  
  def get_stats(self):
    mat = self.df_faces[self.feats_names]
    dists = pairwise_distances(mat)
    df = pd.DataFrame(dists)
    df.columns = self.df_faces[self.NAME_FIELD]
    df.set_index(self.df_faces[self.NAME_FIELD], inplace = True)
    return df


  def show_run_stats(self):
    self.log("FR STATS:\n{}".format(self.get_stats()))  
    self.logger.show_timers()
    return
  

  def _get_distances(self, embed, embed_matrix):
    result = None
    if embed_matrix.shape[0] > 0:
      dists = (embed_matrix - embed)**2
      dists = np.sum(dists, axis = 1)
      dists = np.sqrt(dists)
      result = dists
    return result


  def get_id_vs_all(self, pers_id):
    embed = self.df_faces[self.df_faces[self.ID_FIELD] == pers_id][self.feats_names].values.ravel()
    other_df = self.df_faces[self.df_faces[self.ID_FIELD] != pers_id]
    other_df_short = other_df[[self.ID_FIELD, self.NAME_FIELD]].copy()
    other_embeds = other_df[self.feats_names].values    
    other_df_short.loc[:,'DIST'] = list(self._get_distances(embed, other_embeds))
    return other_df_short

  ##
  ## Face Align zone  
  ##

  def FaceAlign(self, np_img, landmarks, inds = INNER_EYES_AND_BOTTOM_LIP,
                img_dim = 160, scale = 1.0, simple_crop=None):
    """
     tries to align a face based on landmarks
    """
    if self.DEBUG: 
      self.logger.start_timer("    FaceAlign")
    if simple_crop is not None:
      left, top, right, bottom = simple_crop
      thumbnail = np_img[top:bottom,left:right,:].copy()
    else:  
      np_landmarks = np.float32(landmarks)
      np_land_inds = np.array(inds)
  
      # pylint: disable=maybe-no-member
      p1 = np_landmarks[np_land_inds]
      p2 = img_dim * MINMAX_TEMPLATE[np_land_inds] * scale + img_dim * (1 - scale) / 2
      H = cv2.getAffineTransform(p1, p2)
      thumbnail = cv2.warpAffine(np_img, H, (img_dim, img_dim))

    if self.DEBUG: 
      self.logger.end_timer("    FaceAlign")
      self.log("  OmniFR: FaceAlign  in: {}  out:{}".format(np_img.shape,
               thumbnail.shape))
    return thumbnail
  
  
  
  ##
  ## END Face Align zone  
  ##
  
  def GetFaceInfo(self, np_image, get_shape = True, get_id_name = True,
                  get_thumb = True):
    """
     returns a tuple (BOX, SHAPE, EMBED, ID, NAME, IMG, DIST) containing 
     facial info such as:
         BOX - LTRB tuple if face detected in frame or None otherwise
         SHAPE - facial landmarks if get_shape
         EMBED - embedding vector if get_embed
         ID - db user ID if get_id_name
         NAME -  db user name if get_id_name
         IMG - resized/realigned image of the face 
         DIST - distance from the re-identified person         
    """
    _found_box = None
    _shape = None
    _embed = None
    _id = None
    _name = "???"
    _np_resized = None
    _dist = 0
    self.logger.start_timer("  FaceDetect")
    fbox, _found_box = self.face_detect(np_image)    
    self.logger.end_timer("  FaceDetect")
    if get_shape and (fbox != None):
      self.logger.start_timer("  FaceLandmarks")
      landmarks, _shape = self.face_landmarks(np_image, fbox)
      self.logger.start_timer("  FaceLandmarks")
      if landmarks != None:
        
        if get_thumb or self.method == 'keras':
          _np_resized = self.FaceAlign(np_img = np_image, 
                                       landmarks = _shape, 
                                       inds = INNER_EYES_AND_BOTTOM_LIP,
                                       img_dim = self._tf_input_HW[0], 
                                       scale = 1.0)
        if get_id_name:
          fr_res  = self.face_id_maybe_save(np_img = np_image, 
                                            dlib_landmarks = landmarks,
                                            np_box = _found_box,
                                            np_img_resized = _np_resized)
          _id, _name, _embed, _dist = fr_res
    return (_found_box, _shape, _embed, _id, _name, _np_resized, _dist)
       
  
  def _get_current_matrix(self):
    np_matrix = self.df_faces[self.feats_names].values
    return np_matrix
  
  def _save_data(self):
    self.df_faces.to_csv(self.data_file, index = False)
    return
  
  def _find_closest_embedding(self, embed):
    """
     given (NR_EMBEDDINGS,) vector finds closest embedding and returns ID
    """
    result = -1
    min_dist = -1
    np_embeds = self._get_current_matrix()
    if np_embeds.shape[0] > 0:
      dists = (np_embeds - embed)**2
      dists = np.sum(dists, axis = 1)
      dists = np.sqrt(dists)
      min_dist = np.min(dists)
      if min_dist <= self.score_threshold:
        result = np.argmin(dists)      
        if self.DEBUG:
          self.log("       OmniFR: [Idx:{} Dist:{:3f}]".format(
              result, min_dist))
    return result, min_dist
  
  
  def _create_identity(self, embed):
    """
    receives embed and creates new identity in data store
    returns ID and Name
    """
    pers_id = self.df_faces.shape[0] + 10
    pers_name = "PERSOANA_#{}".format(pers_id)
    rec = {}
    rec[self.ID_FIELD] = pers_id
    rec[self.NAME_FIELD] = pers_name
    for i, col in enumerate(self.feats_names):
      rec[col] = embed[i]
      
    self.last_rec = rec
    self.df_faces = self.df_faces.append(rec, ignore_index = True)
    self._save_data()
    if self.DEBUG:
      self.log(" OmniFR: Created new identity {}".format(
          pers_name))
    return pers_id, pers_name


  def _get_name_by_id(self, idpers, use_index = False):
    if use_index:
      sname = self.df_faces.loc[idpers,self.NAME_FIELD]
    else:
      sname = self.df_faces[self.df_faces[self.ID_FIELD]==idpers].loc[0,self.NAME_FIELD]
    return sname

    
  def _get_id_by_index(self, idx):
    return self.df_faces.loc[idx,self.ID_FIELD]
  
  
  def __dl_face_embed(self, np_img, dl_shape):
    return self._dlib_face_recog.compute_face_descriptor(np_img, dl_shape)
  
  def __tf_face_embed(self, np_img, np_ltrb = None):
    """
     np_img MUST be HWC but will be converted to appropriate format
     np_ltrb is LTRB format box
    """
    assert len(np_img.shape) == 3
    if np_ltrb is not None:
      L,T,R,B = np_ltrb
      np_img = np_img[T:B,L:R,:].copy()
    self.log("    OmniFR: Running TF inference on {}".format(np_img.shape))
    
    if np_img.shape[0:2] != self._tf_input_HW:
      np_img = imresize(np_img, self._tf_input_HW)
    
    if self._model_channels == 'channels_first':
      np_img = np_img.T
    
    np_img = np.expand_dims(np_img, axis = 0) / 255
    prev_ch = K.image_data_format()
    K.set_image_data_format(self._model_channels)
    
    if self.method == 'keras':
      lr_phase = K.learning_phase()
    else:
      lr_phase = self._tf_graph.get_tensor_by_name(self._tf_tensors_config["LEARNING_PHASE_TENSOR"])
    embed = self.sess.run(self._tf_fr_output,
                          feed_dict = {self._tf_fr_input : np_img,
                                       lr_phase : 0})
    K.set_image_data_format(prev_ch)
    embed = embed.ravel()
    return embed


  def get_face_embed(self, np_img, dl_shape = None, np_LTRB = None):
    """
     np_img: either full picture or just cropped
     dl_shape: landmarks from dlib
     np_LTRB: left, top, right, bottom used in tf inference preprocessing
    """
    if self.DEBUG:
      self.logger.start_timer("   FaceEMBED")      
      self.log("  OmniFR: {}.FR on {} image".format(self.method, np_img.shape))
    _result = None
    if self.method == 'dlib':
      assert dl_shape != None
      self.logger.start_timer("    FaceEMBED_DLIb")
      _result = self.__dl_face_embed(np_img, dl_shape = dl_shape)
      self.logger.end_timer("    FaceEMBED_DLIb")
    elif self.method in ['tf', 'keras']:
      self.logger.start_timer("    FaceEMBED_TF")
      _result = self.__tf_face_embed(np_img) # np_ltrb = np_LTRB)
      _DEBUG_tf = self.logger.end_timer("    FaceEMBED_TF")
      if self.DEBUG:
        self.log("  OmniFR: {}.FR Time: {:.4f}s".format(self.method,_DEBUG_tf))
        self.logger.end_timer("   FaceEMBED")      
      
    return _result
      
  
  def _get_info(self, embed):
    """
    given generated embedding get ID, Name and L2 distance of proposed embed
    returns -1, "" if not found
    """
    idx, dist = self._find_closest_embedding(embed)
    idpers = -1
    sname = ""
    if idx != -1:
      sname = self._get_name_by_id(idx, use_index = True)
      idpers = self._get_id_by_index(idx)
    return idpers, sname, dist
  
  def face_id_maybe_save(self, np_img, dlib_landmarks, np_box, np_img_resized):
    """
    tries to ID face. Will return ID, Name, Embed if found or new info if NOT found
    also saves new IDs in own face datastore
    must pass np_img (H,W,C) and landmarks_shape (from face_landmarks)
    np_box is LTRB
    """
    if self.DEBUG:  self.logger.start_timer("  FaceID")
    result = (None, None, None)
    # get embed
    img = np_img
    if self.method in ['tf', 'keras']:
      img = np_img_resized
    embed = self.get_face_embed(img, 
                                dl_shape = dlib_landmarks,
                                np_LTRB = np_box)
    if self.DEBUG: self.logger.start_timer("   FaceInfo")
    # try to find if avail
    pers_id, pers_name, dist = self._get_info(embed)
    if pers_id == -1:
      # now create new identity
      pers_id, pers_name = self._create_identity(embed)
    result = (pers_id, pers_name, embed, dist)
    if self.DEBUG: self.logger.end_timer("   FaceInfo")
    if self.DEBUG:  self.logger.end_timer("  FaceID")
    return result
  
  
  def draw_facial_shape(self, np_img, np_facial_shape, left = 0, top = 0):
    self.logger.start_timer("   DrawFacialShape")
    for i in range(np_facial_shape.shape[0]):
      x = int(np_facial_shape[i,0]) + left
      y = int(np_facial_shape[i,1]) + top
      clr = (0,0,255)
      if is_shape(FacialLandmarks.FL_LEYE,i):
        clr = (255,255,255)
      if is_shape(FacialLandmarks.FL_REYE,i):
        clr = (0,255,255)
      np_img = np_circle(np_img, (x, y), 1, clr, -1)
    self.logger.end_timer("   DrawFacialShape")
    return np_img


  
  def face_detect(self, np_img, upsample_detector = 0):
    """
     face detector - will return 1st bounding box both in dlib format and tuple format
     will return None if nothing found
    """
    boxes = self._face_detector(np_img, upsample_detector)
    result = (None, None)
    if len(boxes)>0:
      box = boxes[0]
      LTRB = (box.left(),box.top(),box.right(),box.bottom())
      result = (box, LTRB)
    return result
  
  def multi_face_detect(self, np_img, upsample_detector = 0):
    """
    returns Dlib boxes and LTRB tuples list
     will return None if nothing found
    """
    if self.DEBUG: # half-redundant
      self.logger.start_timer("   FaceDetector")
    boxes = self._face_detector(np_img, upsample_detector)
    if self.DEBUG: # half-redundant
      self.logger.end_timer("   FaceDetector")
    result = (None, None)
    if len(boxes) > 0:
      LTRB_list = []
      for box in boxes:
        LTRB = (box.left(), box.top(), box.right(), box.bottom())
        LTRB_list.append(LTRB)
      result = (boxes, LTRB_list)
    
    return result


  def face_landmarks(self, np_img, dlib_box, large_landmarks = True):
    """
     face landmarks generator - will return numpy array of [points,2] or None if
     nothing found
    """
    result = (None,None)

    if large_landmarks:
      func = self._shape_large_model
      nr_land = 68
    else:
      func = self._shape_small_model
      nr_land = 5
    
    landmarks = func(np_img, dlib_box)
    np_landmarks = np.zeros((nr_land,2))
    for i in range(nr_land):
      np_landmarks[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    result = landmarks, np_landmarks
    
    return result
  
  def GetFaceCropFromLandmarks(self, np_landmarks):
    top = min(np_landmarks[:,1])
    left = min(np_landmarks[:,0])
    right = max(np_landmarks[:,0])
    bottom = max(np_landmarks[:,1])
    
    w = right - left
    h = bottom - top
    
    top -= h * 0.25
    bottom += h * 0.1
    left -= w * 0.15
    right += w * 0.15
    
    return int(left), int(top), int(right), int(bottom)
    
  
  
  def full_scene_process(self, np_img):
    
    SHOW_THUMB = True
    USE_DLIB_LTRB = True

    self.logger.start_timer(" FullSceneProcess")
    np_out_img = np_img.copy()
    
    #first stage detect all faces
    self.logger.start_timer("  MultiFaceDetect")
    boxes, ltrb_list = self.multi_face_detect(np_img)
    self.logger.end_timer("  MultiFaceDetect")
    
    #now process each face !
    if not (boxes is None):
      self.logger.start_timer("  MultiFR")
      self.log(" OmniFR: Processing {} faces".format(len(ltrb_list)))
      for i, LTRB in enumerate(ltrb_list):
        self.log("  OmniFR: Face:{} LTRB:{}".format(i,LTRB))
        box = boxes[i]
        self.logger.start_timer("   MultiFaceLandmarks")
        landmarks, np_shape = self.face_landmarks(np_img, box)
        simple_crop = self.GetFaceCropFromLandmarks(np_shape)
        self.log("  OmniFR: Landmarks LTRB: {}".format(simple_crop))
        
        if USE_DLIB_LTRB:
          simple_crop = LTRB
        
        if min(simple_crop) < 0:
          self.log("  OmniFR: SKIP")
          continue
        self.logger.end_timer("   MultiFaceLandmarks")
        self.logger.start_timer("   MultiFaceID")
        _np_resized = None
        if self.method in ['tf', 'keras']:
          _np_resized = self.FaceAlign(np_img = np_img, 
                                       landmarks = np_shape, 
                                       inds = INNER_EYES_AND_BOTTOM_LIP,
                                       img_dim = self._tf_input_HW[0], 
                                       scale = 1.0,
                                       simple_crop=simple_crop)
        #endif
        _id, _name, _embed, _dist  = self.face_id_maybe_save(np_img, 
                                                             landmarks, 
                                                             LTRB,
                                                             _np_resized)
        _DEBUG_face_id = self.logger.end_timer("   MultiFaceID")
        self.log("  OmniFR: Face [{}/{}/{:.2f}] identified in {:.3f}s".format(
            _id, _name, _dist, _DEBUG_face_id))
        np_out_img = self.draw_facial_shape(np_out_img, np_shape)
        np_out_img = np_rect(LTRB[0], LTRB[1], LTRB[2], LTRB[3], np_out_img,
                           color = (0,255,0), text = _name + ' D:{:.3f}'.format(_dist))        
      # end all faces
      if SHOW_THUMB:
        t_h = _np_resized.shape[0]
        t_w = _np_resized.shape[1]
        np_out_img[:t_h,-t_w:,:] = _np_resized
          
      self.logger.end_timer("  MultiFR")
  
    _DEBUG_full_scene = self.logger.end_timer(" FullSceneProcess")
    if self.DEBUG:
      self.log(" OmniFR Full scene inference and draw time {:.3f}s".format(
          _DEBUG_full_scene))
    return np_out_img
  

  
  
  
if __name__ == '__main__':
  fr_method = 'tf'
  omnifr = FaceEngine(DEBUG = True, fr_method = fr_method)

  vstrm = VideoCameraStream(logger = omnifr.logger,
                            process_func = omnifr.full_scene_process, 
                            info_func = None)  
  if vstrm.video != None:
    video_frame_shape = (vstrm.H,vstrm.W) 
    vstrm.play()
    vstrm.shutdown()

  omnifr.show_run_stats()

  