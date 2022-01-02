# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:57:58 2021

@author: Ultimate LaForsch
"""

import os
import time
import sys


for i in range(1000):
    print('\r' + str(round(i)) + '% complete', end='')
    sys.stdout.flush()


h = 0
s = 0
m = 0

EL = '\x1b[K'  # clear to end of line
CR = '\r'  # carriage return
 
while s <= 60:
    os.system('clear')
    sys.stdout.write(("%d hours %d minutes %s seconds" + EL + CR) % (h, m, s))
    sys.stdout.flush()
    time.sleep(1)
    s += 1
    if s == 60:
        m += 1
        s = 0
    elif m == 60:
        h += 1
        m = 0
        s = 0
        