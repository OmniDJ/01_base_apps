# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 08:38:46 2017

@author: Andrei
"""

from sklearn.datasets import fetch_lfw_people

rect_slice = (slice(70, 195, None), slice(75, 200, None))

lfw_people = fetch_lfw_people(color = True, slice_ = rect_slice) #resize=1, color = True, slice_ = None)