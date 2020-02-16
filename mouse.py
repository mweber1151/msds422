"""
#######################################################
@Filename: mouse.py
@Description: Generates mouse movement to keep active
@Project: N/A
@Project_Lead: N/A
@Author: Mike Weber
@Date Created: Tuesday, Jan 21, 2020
Revisions:
    
    
    
########################################################
"""

"""
#######################################
Import Required Libararies
#######################################
"""
from pynput.mouse import Controller
import time

mouse = Controller()
start = time.time()
while True:
    mouse.move(1,1)
    mouse.move(-1,-1)
    time.sleep(250)
    if time.time() >= (start + 36000):
        break