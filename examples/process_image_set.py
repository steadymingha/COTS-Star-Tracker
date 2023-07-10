#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

used to evaluate ST alg input params and their effect
on solution accuracy and solve times

"""

################################
#LOAD LIBRARIES
################################
import os
import cv2
import csv
import json
import time
import psutil
import subprocess
import numpy as np
from datetime import datetime
from star_tracker import main
from star_tracker.cam_matrix import *
from star_tracker.array_transformations import *

################################
#USER INPUT
################################
nmatch = 6 # minimum number of stars to match
starMatchPixelTol = 2 # pixel match tolerance
min_star_area = 5 # minimum pixel area for a star
max_star_area = 20000 # maximum pixel area for a star
max_num_stars_to_process = 20 # maximum number of centroids to attempt to match per image

low_thresh_pxl_intensity = None
hi_thresh_pxl_intensity = None

VERBOSE = True # set True for prints on results
graphics = False # set True for graphics throughout the solve process


data_path = '../test/test_single' # path to your data
data_path = 'C:/Users/mhlee/Desktop/Study/FSO_centroid/img_gen' # path to your data
cam_config_file = '../data/cam_config/xic_ximea_cam_example.json' # the name of your camera config file, assumed to be in the data/camera_cal directory
darkframe_file = '' # the name of your darkfile file, assumed to be in the data_path directory
image_extension = ".jpg" # the image extension to search for in the data_path directory



################################
#SUPPORT FUNCTIONS
################################



################################
#MAIN CODE
################################

#load star tracker stuff
cam_cal_dir = data_path
input_data_dir = data_path

if darkframe_file == '': darkframe_file = None
if darkframe_file is not None:
    darkframe_file = os.path.join(data_path, darkframe_file)
    if not os.path.exists(darkframe_file): darkframe_file = None
    print("darkframe file (None means unable to find): " + darkframe_file)

k = None # np.load(os.path.join(data_path, 'k.npy'))
m = None # np.load(os.path.join(data_path, 'm.npy'))
q = None # np.load(os.path.join(data_path, 'q.npy'))
x_cat = None # np.load(os.path.join(data_path, 'u.npy'))
indexed_star_pairs = None # np.load(os.path.join(data_path, 'indexed_star_pairs.npy'))

cam_file = '../data/cam_config/xic_ximea_cam_example.json'#os.path.join(cam_cal_dir, cam_config_file)
# camera_matrix, _, _ = read_cam_json(cam_file)
# dx = camera_matrix[0, 0]
# isa_thresh = starMatchPixelTol*(1/dx)

#define structures for data capture
image_name = []
ttime = []
stemp = []
sram  = []
scpu  = []
solve_time = []
qs = []
qv0 = []
qv1 = []
qv2 = []

# create list of all images in target dir
total_start = time.time()
dir_contents = os.listdir(data_path)
image_names = []

for item in dir_contents:
    if image_extension in item:
        image_names+=[os.path.join(os.path.abspath(data_path),item)]

for image_filename in image_names:

    image_name += [image_filename]
    print("===================================================")
    print(image_filename)

    #run star tracker
    solve_start_time = time.time()

    q_est, idmatch, nmatches, x_obs, rtrnd_img = main.star_tracker(
            image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file,
            min_star_area=min_star_area, max_star_area=max_star_area, nmatch=nmatch, n_stars=max_num_stars_to_process,
            low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE)

        # image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file,
            # min_star_area=min_star_area, max_star_area=max_star_area, isa_thresh=isa_thresh, nmatch=nmatch, n_stars=max_num_stars_to_process,
            # low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE)


#     solve_time += [time.time()-solve_start_time]
#
#     #collect data
#     try:
#         assert not np.any(np.isnan(q_est))
#         if VERBOSE:
#             print('est q: ' + str(q_est)+'\n')
#         qs += [q_est[3]]
#         qv0 += [q_est[0]]
#         qv1 += [q_est[1]]
#         qv2 += [q_est[2]]
#     except AssertionError:
#         if VERBOSE:
#             print('NO VALID STARS FOUND\n')
#         qs += [999]
#         qv0 += [999]
#         qv1 += [999]
#         qv2 += [999]
#
#
#
#     ttime += [time.time()]
#     sram  += [psutil.virtual_memory().percent]
#     #scpu  += [psutil.cpu_percent(2)]
#     scpu  += [psutil.cpu_percent()]
#
#
# data = {'image name':image_name,'time':ttime,'RAM':sram,'CPU':scpu,'image solve time (s)':solve_time, 'qs':qs,'qv0':qv0,'qv1':qv1,'qv2':qv2}
#
# now = str(datetime.now())
# now = now.split('.')
# now = now[0]
# now = now.replace(' ','_')
# now = now.replace(':','-')
#
# #write stuff
# keys=sorted(data.keys())
# filename = now+'_data_file_nm-'+str(nmatch)+'_pxl-'+str(starMatchPixelTol)+'.csv'
# with open(filename,'w', newline='') as csv_file:
#              writer=csv.writer(csv_file)
#              writer.writerow(keys)
#              writer.writerows(zip(*[data[key] for  key in keys]))
#
# print("\n\n took " + str(time.time()-total_start) + " seconds to complete \n\n")
# print("data saved to: " +filename)
#
# print("\n\nTHE END\n\n")





