from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
import cPickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math
from common import *

def is_chinese(ch): 
    #uc=ch.decode('utf-8')        
    if u'\u4e00' <= ch<=u'\u9fff':
        return True
    else:
        return False
        
txt_source='/home/yuz/lijiahui/syntheticdata/SynthText/data/newsgroup/chinese_txt_source.txt'
f=open(txt_source,'r')
for line in f.readlines():
    print line
    for ch in line.decode('utf-8'):
        print is_chinese(ch) or ch.isalnum()
