# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:31:22 2019

@author: Animesh
"""

import os, sys
from shutil import copyfile
import glob
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]



first_loc='K:\Data_Process_OCD\T1Img\CONT\Org'
second_loc='K:\\Data_Process_OCD\\SCZ_OCD_ML\\Resting_State\\Controls_NIMHANS\\0.01-0.08Hz\\FunImgARCWSF'


dirlist =  os.listdir(first_loc)

Newdirlist1=[]

for dir_Name in dirlist:
    positions= find(dir_Name,'_')
    Newdirlist1.append(dir_Name[2:positions[0]])
    
print (Newdirlist1)   



dirlist =  os.listdir(second_loc)

Newdirlist2=[]

for dir_Name in dirlist:
    positions= find(dir_Name,'_')
    Newdirlist2.append(dir_Name[0:positions[0]])
    
print (Newdirlist2)   



def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 




#dirlist.remove('rename.py')
#dirlist.remove('desktop.ini')
print (dirlist)
