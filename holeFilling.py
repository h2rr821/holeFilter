# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:42:25 2019

@author: h2r
"""

import cv2
import numpy as np

def main():
   
    img= cv2.imread('images.png',0) 
    ret, bimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("org",bimg)
    
    
    #1. Form a mask of image,invert the image========================================
    mask=~bimg
    #cv2.imshow("mask",mask)
    
    #2.form masker of image=========================================================
    bw = 1  #width of border required
    temp = 255*np.ones(bimg.shape[:2], dtype = "uint8") 
    temp = cv2.rectangle(temp, (bw,bw),(bimg.shape[1]-bw,bimg.shape[0]-bw), 0, -1) 
    marker = cv2.bitwise_and(mask,temp)

    #cv2.imshow("marker",marker)
    
    
    #3. Perform the morphological reconstruction==================================
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    res=cv2.dilate(marker,kernel, iterations=1)
    res=cv2.min(res,mask)
    #cv2.imshow("res",res)
    res_old=temp
    while not np.array_equal(res,res_old):
        res_old=res
        res=cv2.min(cv2.dilate(res_old, kernel,iterations=1),mask)
    
    #4. obtain the complement
    final=~res
    
    cv2.imshow("final",final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ =='__main__':
    main()