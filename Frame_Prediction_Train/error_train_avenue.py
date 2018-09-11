import sys
sys.path.insert(0,'/opt/caffe/python')
import numpy as np
import caffe
import h5py
import os
import glob
import fnmatch
from PIL import Image
import scipy.misc
import scipy.io

caffe.set_device(0)
caffe.set_mode_gpu()

frame_path='/nobackup/schtmt/ucsd_data/AvenueAppearance_10x1x227x227/'
save_path = '/nobackup/schtmt/model_eval'
if not os.path.exists(save_path):
        os.mkdir(save_path)        
eval_proto = 'eval.prototxt'
model_path = '/nobackup/schtmt/conv_wta_lstm_model/tanhLSTM_prediction_avenue'
models = glob.glob(model_path + '/*.caffemodel')
num_model = len(models)
print(num_model)

eval_error = np.zeros((1,num_model))

for i in range(num_model):#(num_model):
    cur_model = os.path.join(model_path,'_iter_%d.caffemodel') % ((i+1)*1000)
    net = caffe.Net(eval_proto,
                    cur_model, 
                    caffe.TEST)          
    
    err_acc = 0
    num_eval_data = 0
#    cur_err = np.zeros((1,num_iters))
    for seq in range(44):
	eva_file = h5py.File(os.path.join(frame_path,'train_vol_%d.h5'%(seq+1)),'r')
	eva_data = eva_file['input'][:]
	num_data = np.shape(eva_data)[0]
	num_iters = num_data / 8 
    
        for iter_id in range(num_iters):
            print(iter_id)
            cur_data = eva_data[8*iter_id:8*(iter_id+1),:,:,:,:]
            net.blobs['input'].data[...] = cur_data
            out = net.forward()
            err_acc += 2*8*net.blobs['recons_error'].data
            num_eval_data += cur_data.shape[0]
    eval_error[:,i] = err_acc / (2*num_eval_data)        
#            prediction = net.blobs['deconv3'].data # 40x1x227x227
#            prediction = np.reshape(prediction,[5,8,1,227,227])
#            prediction = np.transpose(prediction,(1,0,2,3,4))
        #print(prediction.shape)
#            err = np.power((cur_data[:,5:10,:,:,:] - prediction),2)   
#            err = np.reshape(err,[8*5,227*227])
#            err = np.sqrt(np.sum(err,1))
        #print(err/8)
#            err_acc += np.sum(err) 
#            num_eval_data += cur_data.shape[0]
#	cur_err[:,iter_id] = np.sum(err)
#        error[i,iter_id] = np.sum(err)
	
    #print(err_acc)
   # print(num_eval_data)

#    eval_error[:,i] = err_acc / num_eval_data
    
#np.save(os.path.join(save_path,'5000_5.npy'),error)    
    np.save(os.path.join(save_path,'tanhLSTM_prediction_avenue_train_mse.npy'),eval_error)    
