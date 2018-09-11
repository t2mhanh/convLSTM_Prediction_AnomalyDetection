import sys
sys.path.insert(0,'/opt/caffe/python')
import numpy as np
import caffe
import h5py
import os
import glob
import fnmatch
from PIL import Image
import time

#caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_model_cpu()

frame_path='/nobackup/schtmt/ucsd_data/Ped1_images/Test/'
save_path = '/nobackup/schtmt/model_gridSearch/prediction6_aug2_ped1'
if not os.path.exists(save_path):
        os.mkdir(save_path)
        
train_proto = 'test2.prototxt'
model_path = '/nobackup/schtmt/conv_wta_lstm_model/convLSTM_prediction6_ped1'

cur_model = os.path.join(model_path,'_iter_360000.caffemodel')
net = caffe.Net(train_proto,
                cur_model, 
                caffe.TEST) 
total_time = 0
total_frame = 0
for seq in range(36):
    print(seq)    
    # load images
    if seq < 9:                
        dirpath = (frame_path + 'Test00%d' % (seq+1))
    else:
        dirpath = (frame_path + 'Test0%d' % (seq+1))
    nfr = len(fnmatch.filter(os.listdir(dirpath), '*.tif'))

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
    start_time = time.clock()	
    for fr in range(nfr-9):
        err = []
        cur_vol = frame_seq[fr:fr+10,:,:]
        cur_vol = cur_vol.reshape([1,10,1,227,227])
        net.blobs['input'].data[...] = cur_vol
        out = net.forward()        
        prediction = net.blobs['deconv3'].data
        prediction = np.reshape(prediction,[5,227,227])
        err = np.power((np.reshape(cur_vol[:,5:10,:,:,:],[5,227,227]) - prediction),2)   
        err = np.reshape(err,[5,227*227])
        err = np.sqrt(np.sum(err,1))
        frame_error[:,fr] = np.sum(err)
    total_time = total_time + time.clock() - start_time
    total_frame = total_frame + nfr
    print(total_time,total_frame)
print("Total times: ", total_time, "second")        
print("Total frames:", total_frame)
print("Processing time:", total_time/total_frame, "second") 

