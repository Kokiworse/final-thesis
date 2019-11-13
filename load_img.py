# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:31:48 2018

@author: koki
"""

import os
from PIL import Image

def load_images():
    imgs = []
    rootDir = "C:\\Users\\koki\\Desktop\\img"
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:


            s = "{dirName}\\{fname}".format(dirName = dirName, fname = fname )
            # return only the imgs
            if s[ -3 : ] != "jpg":
                continue
            imgs.append(s)
    return imgs

if __name__ == "__main__":
    load_images()
