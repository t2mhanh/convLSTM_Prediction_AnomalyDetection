
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_Sep_anaconda/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
import h5py
import cv2
import os
import time
import itertools as it

caffe.set_device(0)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

# aug1
# net = caffe.Net('test_input5frames.prototxt',
#                 '/usr/not-backed-up/MODELS_DATA/models/ped1_iter_360000.caffemodel', 
#                 caffe.TEST) 

# aug2
net = caffe.Net('test_input5frames.prototxt',
                '../convLSTM_models/ped1_aug2.caffemodel', 
                caffe.TEST) 


# In[2]:

import fnmatch
import os
from PIL import Image
frame_path='/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/'
# save_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction6/' # aug1
save_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction_aug2/' # aug2
if not os.path.exists(save_path):
    os.mkdir(save_path)
pred_time = 0
processing_time = 0
total_frame = 0
for seq in range(36):
    print(seq)    
    # load images
    if seq < 9:                
        dirpath = (frame_path + 'Test00%d' % (seq+1))
    else:
        dirpath = (frame_path + 'Test0%d' % (seq+1))
    nfr = len(fnmatch.filter(os.listdir(dirpath), '*.tif'))
   
    
    frame_error = np.zeros([1,nfr-9])                    
    for fr in range(nfr-9):   
        start_time = time.clock()
        if fr == 0:
            frame_seq = np.zeros((10,227,227))
            for idx in range(10):
                if (fr+1+idx) < 10:
                    fr_id = '00%d' % (fr+1+idx)                                       
                elif (fr+1+idx) < 100:
                    fr_id = '0%d' % (fr+1+idx)                
                else:
                    fr_id = '%d' % (fr+1+idx) 
                im1 = Image.open(dirpath + '/' + (fr_id) + '.tif')
                im1 = np.array(im1.resize((227,227),Image.BILINEAR))#size [width,height]            
                im1 = np.reshape(im1,[1,227,227])
                im1 = im1.astype(float) / 255.
                frame_seq[idx,:,:] = im1        
        else:
            if (fr+10) < 10:
                fr_id = '00%d' % (fr+10)                                       
            elif (fr+10) < 100:
                fr_id = '0%d' % (fr+10)                
            else:
                fr_id = '%d' % (fr+10) 
            im1 = Image.open(dirpath + '/' + (fr_id) + '.tif')
            im1 = np.array(im1.resize((227,227),Image.BILINEAR))#size [width,height]            
            im1 = np.reshape(im1,[1,227,227])
            im1 = im1.astype(float) / 255.
            
            frame_seq = np.append(frame_seq[1:10,:,:],im1,0)   
        processing_time = processing_time + time.clock() - start_time
        
        start_time_pred = time.clock()
        net.blobs['input'].data[...] = np.reshape(frame_seq[0:5,:,:],[1,5,1,227,227])
        out = net.forward()        
        prediction = net.blobs['deconv3'].data
        prediction = np.reshape(prediction,[5,227,227])
        err = np.power((frame_seq[5:10,:,:] - prediction),2)                
        # error of 5 frames
        err = np.reshape(err,[5,227*227])
        err = np.sqrt(np.sum(err,1))
        frame_error[:,fr] = np.sum(err)
        pred_time = pred_time + time.clock() - start_time_pred
#         print(exec_time)
#         total_time = total_time + exec_time
#     total_time = total_time + time.clock() - start_time
    total_frame = total_frame + nfr
    filename = ("test_%d_error.h5" % (seq+1))        

    h5f = h5py.File(os.path.join(save_path,filename), 'w')
    h5f.create_dataset('frame_error', data=frame_error)                
    h5f.close()   
      
print("Total frames:", total_frame)
print("Processing time:", processing_time/total_frame, "second")        
print("Processing time:", pred_time/total_frame, "second")        

