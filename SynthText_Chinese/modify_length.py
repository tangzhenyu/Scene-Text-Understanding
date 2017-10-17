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
from random import randint 
rp_path='/home/yuz/lijiahui/syntheticdata/SynthText/data/newgroup_modify/'
fn='/home/yuz/lijiahui/syntheticdata/SynthText/data/newsgroup/'

files= os.listdir(fn)
files=files[0:-1]
for filename in files:
    print filename   
    fc=filename#.decode('utf-8')
    write_path=rp_path+fc
    fc=fn+fc
    print fc
    with open(fc,'r') as f:
        f2=open(write_path,'w')
        print write_path
        for l in f.readlines():
            line=l.strip()
            #line=line.decode('utf-8')
            print line
            for i in range(len(line)):
                tl=randint(0,9)
                if i + tl<len(line):
                    f2.write('%s\n'%line[i:i+tl])
                else:
                    f2.write('%s\n'%line[i:-1])  
