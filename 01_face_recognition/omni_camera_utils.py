# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:43:11 2017

@author: Andrei
"""


import cv2
import numpy as np

from PIL import Image, ImageDraw
from time import sleep


__version__ = "0.2.cv2"
__author__  = "Andrei Ionut Damian"
__copyright__ = "(C) Knowledge Investment Group"
__project__ = "OmniDJ"

def test_func(np_image):
  y1 = np_image.shape[0]//4
  x1 = np_image.shape[1]//4
  y2 = y1 * 3
  x2 = x1 * 3
  np_image[y1,x1:x2,:] = [255,255,255]
  np_image[y2,x1:x2,:] = [255,255,255]
  np_image[y1:y2,x1,:] = [255,255,255]
  np_image[y1:y2,x2,:] = [255,255,255]
  return np_image

def np_circle(np_img, center, radius, color = (255, 255, 255), thickness = -1):
  np_img = cv2.circle(np_img, center, radius, color, thickness)
  return np_img

def _np_rect(left,top,right,bottom, np_img, color = (255,255,255), thickness = 1):

  seg_h = int((bottom - top) * 0.2)
  seg_w = int((right - left) * 0.2)
  np_img[top, left:(left+seg_w) ,:] = color
  np_img[top, (right-seg_w):right, :] = color
  np_img[bottom, left:(left+seg_w), :] = color
  np_img[bottom, (right-seg_w):right, :] = color
  np_img[top:(top+seg_h), left, :] = color
  np_img[(bottom-seg_h):bottom, left, :] = color
  np_img[top:(top+seg_h), right, :] = color
  np_img[(bottom-seg_h):bottom, right, :] = color

  if thickness>1:
    np_img[top+1, left:(left+seg_w) ,:] = color
    np_img[top+1, (right-seg_w):right, :] = color
    np_img[bottom-1, left:(left+seg_w), :] = color
    np_img[bottom-1, (right-seg_w):right, :] = color
    np_img[top:(top+seg_h), left+1, :] = color
    np_img[(bottom-seg_h):bottom, left+1, :] = color
    np_img[top:(top+seg_h), right-1, :] = color
    np_img[(bottom-seg_h):bottom, right-1, :] = color
  return np_img
  

def np_rect(left, top, right, bottom, np_img, cls_, color = (255,255,255), 
            thickness = 0.4, text = None, use_cv2 = False, font_size=0.5):
  """
  draws "target" rect on box 
  np_image must be (H,W,C)
  use_cv2 = True for fast drawing !
  """
  color = tuple(color)
  left = max(0, int(left))
  top = max(0, int(top))
  right = min(np_img.shape[1]-1, int(right))
  bottom = min(np_img.shape[0]-1, int(bottom))

  if use_cv2:
    np_img = cv2.rectangle(np_img, pt1=(left,top), pt2=(right, bottom), 
                           color=color, thickness=thickness)

  else:
    np_img = _np_rect(left,top,right,bottom, np_img, color = color, thickness=thickness)

  return np_img

class VideoCameraStream:
  def __init__(self, logger = None, file_name = None,
               process_func = None, info_func = None, 
               onclick_func = None, hd=1, camera=0):
    self.process_func = process_func
    self.onclick_func = onclick_func
    self.info_func = info_func
    self.__version__ = __version__
    self.logger = logger
    self.log("Initializing VideoCameraStream v.{}".format(self.__version__))
    if file_name == None:
      self.video = cv2.VideoCapture(camera)
      if hd==1:
        std_h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        std_w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.log("Standard resolution W:{} H:{}".format(std_w,std_h))
        new_h = 720
        new_w = 1280
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
        cur_h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cur_w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.log("New resolution W:{} H:{}".format(cur_w,cur_h))
      success, frame = self.video.read()
      self.win_name = "frame_win"
      cv2.namedWindow(self.win_name)
      cv2.moveWindow(self.win_name, 1, 1)
      cv2.setMouseCallback(self.win_name, self.click_event)
      if not success:
        if self.logger != None:
          self.log("Problem capturing video stream...")
        self.video = None
      else:
        if self.logger != None:
          self.logger.VerboseLog("Captured video stream {}".format(frame.shape))
      self.H = frame.shape[0]
      self.W = frame.shape[1]
    else:
      self.video = cv2.VideoCapture(file_name)
    return
  
  def log(self,strmsg):
    if self.logger != None:
      self.logger.VerboseLog(strmsg)
    else:
      print(strmsg, flush = True)
    return

  def __del__(self):
    if self.video != None:
      self.video.release()
    cv2.destroyAllWindows()
    
  def get_frames(self):
    _, frame = self.video.read()
    _, jpeg = cv2.imencode('.jpg',frame)
    return jpeg.tobytes()
  
  def play(self, sleep_time = 0, ):
    
    assert self.video != None    
    total_errors = 0
    while(True):
      success, frame = self.video.read()      
      if success:
        if sleep_time>0:
          sleep(sleep_time)
        if self.process_func != None:
          out_frame = self.process_func(frame)
        else:
          out_frame = frame
        cv2.imshow(self.win_name,out_frame)
      else:
        self.log("Error in OpenCV video stream!")
        total_errors += 1
        sleep(1)
      
      key = cv2.waitKey(1) & 0xFF
      if (key == ord('q')) or (total_errors>2) or (key == ord('Q')):
          break
      else:
        skey = chr(key)
        skey = skey.upper()
        if key>=65 and key<=122: # if normal ascii
          self.info_func(skey, img = out_frame)
        
    cv2.destroyAllWindows()
    return
  
  def shutdown(self):
    cv2.destroyAllWindows()
    self.video.release()
    self.video = None
    return
    

  def click_event(self, event, x, y, flags, param):  
    if event == cv2.EVENT_LBUTTONDOWN:
      self.log("  OmniCamera click ({},{}) {} {}".format(x, y, flags, param))
    if self.onclick_func is not None:
      self.onclick_func(x,y)
    return
    
    

if __name__ == '__main__':
  vcap = VideoCameraStream()
  vcap.play()
  #vcap.play(1, process_func = test_func)
  vcap.shutdown()
  