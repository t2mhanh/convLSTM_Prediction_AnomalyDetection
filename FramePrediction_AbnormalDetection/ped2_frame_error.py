
# coding: utf-8

# In[3]:

import sys
sys.path.insert(0,'/home/csunix/schtmt/NewFolder/caffe_Sep_anaconda/python')
import numpy as np
import matplotlib.pyplot as plt
import caffe
import h5py
import cv2
import os
from numpy import prod, sum

caffe.set_device(0)
caffe.set_mode_gpu()

# aug1
net = caffe.Net('test_input5frames.prototxt',
                '../convLSTM_models/ped2_aug1_epoch80.caffemodel', 
                caffe.TEST)


# In[5]:

#PED2
import fnmatch
import os
from PIL import Image
frame_path='/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
# aug1
# save_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction16_ped2/'#icip paper
save_path = '/usr/not-backed-up/1_convlstm/prediction_aug1_ped2/'

#aug2
# save_path = '/usr/not-backed-up/1_convlstm/prediction_aug2_ped2/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for seq in range(12):
    print(seq)    
    # load images
    if seq < 9:                
        dirpath = (frame_path + 'Test00%d' % (seq+1))
    else:
        dirpath = (frame_path + 'Test0%d' % (seq+1))
    nfr = len(fnmatch.filter(os.listdir(dirpath), '*.tif'))
#         frame_seq = np.zeros((227,227,nfr),dtype=np.float64)
    for fr in range(nfr):            
        if fr < 9:
            fr_id = '00%d' % (fr+1)                                       
        elif fr < 99:
            fr_id = '0%d' % (fr+1)                
        else:
            fr_id = '%d' % (fr+1)                
        im1 = Image.open(dirpath + '/' + fr_id + '.tif')
        im1 = np.array(im1.resize((227,227),Image.BILINEAR))#size [width,height]            
        im1 = np.reshape(im1,[1,227,227])
        im1 = im1.astype(float) / 255.
        if fr == 0:
            frame_seq = im1
        else:
            frame_seq = np.append(frame_seq,im1,0)
    frame_error = np.zeros([1,nfr-9])
    print(frame_error.shape)
    for fr in range(nfr-9):
        err = []
        cur_vol = frame_seq[fr:fr+10,:,:]        
        net.blobs['input'].data[...] = np.reshape(cur_vol[0:5,:,:],[1,5,1,227,227])
        out = net.forward()        
        prediction = net.blobs['deconv3'].data
        prediction = np.reshape(prediction,[5,227,227])
        err = np.power((cur_vol[5:10,:,:] - prediction),2)
        
        
#         # because of nummerical error with caffe
#         input_ = net.blobs['input'].data        
#         err = np.power((np.reshape(input_[:,5:10,:,:,:],[5,227,227]) - prediction),2)
                
        # error of 5 frames
        err = np.reshape(err,[5,227*227])
        err = np.sqrt(np.sum(err,1))
        frame_error[:,fr] = np.sum(err)   
    filename = ("test_%d_error.h5" % (seq+1))       
    h5f = h5py.File(save_path + filename, 'w')
    h5f.create_dataset('frame_error', data=frame_error)                
    h5f.close()


# In[7]:

print(err)
print(input_.shape)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(0,frame_error.shape[1]).reshape(1,frame_error.shape[1])[0,:],frame_error[0,:],'r-')
plt.show()
# 24500

