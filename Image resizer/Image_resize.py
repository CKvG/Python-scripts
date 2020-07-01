# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:21:31 2020

@author: grundch
"""


from PIL import Image
import os

DIR = os.getcwd()

ALL_FILES = True
if not ALL_FILES: FILESIZE = 102335 

if ALL_FILES:
    files = [f for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))]
else:
    files = [f for f in os.listdir(DIR) \
                     if os.path.isfile(os.path.join(DIR, f)) \
                         and os.path.getsize(os.path.join(DIR, f)) > FILESIZE]

for f in files:
    foo = Image.open(DIR + '\\' + f)
    #foo = foo.resize((160,300),Image.ANTIALIAS)
    foo.save(DIR + '\\' + f, optimize=True, quality=85)
