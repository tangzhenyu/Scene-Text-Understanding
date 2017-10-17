#coding:utf-8
import os
import pickle
import sys
import codecs
import time
import random
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np
from math import *
import numpy.ma as ma

def r(val):
    return int(np.random.random() * val)
def random_scale(x,y):
    ''' 对x随机scale,生成x-y之间的一个数'''
    gray_out = r(y+1-x) + x
    return gray_out

def text_Gengray(bg_gray, line):
    gray_flag = np.random.randint(2)
    if bg_gray < line:
        text_gray = random_scale(bg_gray + line, 255)
    elif bg_gray > (255 - line):
        text_gray = random_scale(0, bg_gray - line)
    else:
        text_gray = gray_flag*random_scale(0, bg_gray - line) + (1 - gray_flag)*random_scale(bg_gray+line, 255)
    return text_gray

def GenCh(f,val, data_shape1, data_shape2, bg_gray, text_gray, text_position):
    img=Image.new("L", (data_shape1,data_shape2),bg_gray)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),val,text_gray,font=f)
    #draw.text((0, text_position),val.decode('utf-8'),0,font=f)
    A = np.array(img)

    #二值化,确定文字精确的左右边界
    if bg_gray > text_gray:
	ret,bin = cv2.threshold(A,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
	ret,bin = cv2.threshold(A,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('A',A)
    #cv2.imshow('bin',bin)


    left = -1
    right = 10000
    for i in range(0,bin.shape[1]):
	if np.sum(bin[:,i]) > 0:
		left = i
		break
    for i in range(bin.shape[1]-1,0,-1):
	if np.sum(bin[:,i]) > 0:
		right = i
		break
    dst  = A[:,left:right+1]
    #cv2.imshow('dst',dst)
    #cv2.waitKey()
    return dst
def tfactor(img):

    img[:,:] = img[:,:]*(0.8+ np.random.random()*0.2)
    return img
def Addblur(img, val):
    blur_kernel = random_scale(2,val)
    #print blur_kernel
    img = cv2.blur(img, (blur_kernel,blur_kernel))
    return img
def motionBlur(img,val):
    blur_kernel0 = random_scale(2,val)
    blur_kernel1 = random_scale(2,val)
    anchor = (random_scale(0,blur_kernel0-1),random_scale(0,blur_kernel1-1))
    img = cv2.blur(img,(blur_kernel0,blur_kernel1),anchor=anchor)
    return img
def AddNoiseSingleChannel(single):
    diff = (255-single.max())/3    
    noise = np.random.normal(0,1+r(6),single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst
def rot(img,angel,shape,max_angel,bg_gray):
    size_o = [shape[1],shape[0]]

    size = (shape[1] + int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])


    interval = abs(int(sin((float(angel) /180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0], [0,size_o[1]], [size_o[0],0], [size_o[0], size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size,borderValue=bg_gray)

    return dst

def rotRandrom(img, factor, size, bg_gray):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size, borderValue=bg_gray)
    return dst

class GenText:
    def __init__(self, ch_size=16,imgHeight=16,imgWidth=64):
	self.ch_size  = ch_size
	self.imgHeight = imgHeight
	self.imgWidth = imgWidth

    def draw(self,val,font):
        bg_gray = r(256) #随机生成背景灰度
	#bg_gray = 0
        text_gray = text_Gengray(bg_gray, 60)#生成前景灰度
	#text_gray = random_scale(30,256)
        text_position = random_scale(0,(self.imgHeight-self.ch_size)/2) #垂直方向文本位置
	print 'text_pos: ',text_position
        offset_left = int(np.random.random() * self.ch_size)
        
        offset = offset_left
	ch_num = len(val)
	imgWidth = min(self.imgWidth,offset+ch_num*self.ch_size)
        img = np.array(Image.new("L", (imgWidth, self.imgHeight), bg_gray))
        base = offset_left

	#间距
	inter = random.randint(1,5)
	print inter
	writeTxt = ''
        for i in range(ch_num):
	    if (base+self.ch_size) <= imgWidth:
		tmp = GenCh(font,val[i], self.ch_size, self.imgHeight, bg_gray, text_gray, text_position)
	        img[0: self.imgHeight, base : base + tmp.shape[1]]= tmp
                base += tmp.shape[1]+inter
		writeTxt += val[i]
	    else:
		break
        return img, bg_gray,text_gray,writeTxt
    def changeBG(self,input,fg_gray,bgImg):
	assert len(bgImg.shape) == 2
	if bgImg.shape[0] < input.shape[0] or bgImg.shape[1] < input.shape[1]:
		return input
	thresh = 40
	if bgImg.shape[0]-input.shape[0]-1 <= 0 or bgImg.shape[1]-input.shape[1]-1<=0:
		return input
			
	st_y = random.randint(0,bgImg.shape[0]-input.shape[0]-1)
	st_x = random.randint(0,bgImg.shape[1]-input.shape[1]-1)
	#bgImg
	tmp = bgImg[st_y:st_y+input.shape[0],st_x:st_x+input.shape[1]]

	mean = np.mean(tmp)
	if abs(fg_gray - mean) < thresh:
		return input
	else:
		output = tmp.copy() 
		
	h = input.shape[0]
	w = input.shape[1]
	ret,input = cv2.threshold(input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	output[input>0] = fg_gray
	return output
	
	
    def generate(self,text,font):
        fg, bg_gray,fg_gray,txt = self.draw(text,font)
        com = rot(fg,r(90)-45,fg.shape,45, bg_gray)

	#更换背景图片
	#com = self.changeBG(com,fg_gray,bgImg)
	#com = rotRandrom(fg,2,(fg.shape[1],fg.shape[0]), bg_gray)
        #com = tfactor(com)
	com = motionBlur(com,2)
        com = AddNoiseSingleChannel(com)
	if com.shape[1] < self.imgWidth:
	    tmp = np.zeros((self.imgHeight,self.imgWidth),dtype='uint8')
	    tmp[:,:] = 128
	    tmp[:,0:com.shape[1]] = com.copy()
	    com = tmp.copy()
        elif com.shape[1] > self.imgWidth: #rot时，可能会宽度增加一点
	    com = cv2.resize(com,(self.imgWidth,self.imgHeight))
	    
        return com,txt


 
  
def genTextImg():
        
	num = 10000
	maxNum = 12
	Gs = []
	fonts = []
	#font_sizes = [26,27,28,29,30]
	font_sizes = [11,12,13,14,15,16]
	for size in font_sizes:
		Gs.append(GenText(size,16,64))
		tmp = []
		tmp.append(ImageFont.truetype('./font/仿宋_GB2312.ttf',size))
		tmp.append(ImageFont.truetype('./font/华文隶书.TTF',size))
		tmp.append(ImageFont.truetype('./font/宋体_GB18030+%26+新宋体_GB18030.ttc',size))
		tmp.append(ImageFont.truetype('./font/微软vista黑体.ttf',size))
		tmp.append(ImageFont.truetype('./font/方正楷体GBK.ttf',size))
		tmp.append(ImageFont.truetype('./font/方正隶书简体.ttf',size))
		tmp.append(ImageFont.truetype('./font/楷体_GB2312.ttf',size))
		tmp.append(ImageFont.truetype('./font/造字工房尚黑G0v1纤细长体.otf',size))
		fonts.append(tmp)

	outputPath = 'data/train/'
	txtPath = "corpus/train/"
        txtFiles = os.listdir(txtPath)
        index=0
        for file in txtFiles:
            fullPath = txtPath + file
            ##输入文档
            with open(fullPath, "r") as f: 
                    content = f.readlines()
                    f.close()

            #index
            files = os.listdir(outputPath)
            #index = len(files) + 1
            for txt in content:
                    txt = txt.strip()
                    unicode1 =  txt.decode('utf-8')
                    if unicode1 == u"\n":
                            continue
                    flag = random.randint(1,10) <= 9 #写8个字的概率 0.8
                    #if flag:
                    #        count = maxNum
                    #else:
                    #        count = random.randint(2,maxNum-1)
                    count=int(index/1000)+1      
                    lines = [unicode1[i:i+count+10] for i in range(0, len(unicode1), count+10)] # 
                    for line in lines:
                            newline = ''
                            for ch in line:
                                if True and ord(ch)!=12288:
                                    newline += ch #字符如果在我的字库中，我才生成图像
                                newline=newline.replace(" ","")
                                if len(newline)==count:
                                    break    
                            
                            if len(newline)<count:
                                    continue

                            index += 1
                            print index,newline,len(newline)
                            filename =  str(index) + ".jpg"
                            #writePath =outputPath +'/'+str(len(newline))+'/'+ filename
                            writePath ='/home/yuz/lijiahui/ocr/background_judge/sentencedata/1/ex_' +filename
                            Gid = random.randint(0,len(Gs)-1)
                            fontid = random.randint(0,len(fonts[Gid])-1)
                            #bgImg=cv2.imread('data/train/2011060409214653.jpg',1)
                            img,res_txt = Gs[Gid].generate(newline,fonts[Gid][fontid])
                            cv2.imwrite(writePath,img)
                            if index >= num:
                                break
                    if index >= num:
                        break
        #fin.close()

if __name__ == '__main__':
	genTextImg()
	
	




