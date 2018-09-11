# convLSTM_Prediction_AnomalyDetection
We combine reconstruction/prediction over many frames with interleaved LSTMs to train end-to-end deep models 
for learning normal appearances and motions in training video sequences. Then deriving an anomaly score from 
reconstruction/prediction error of testing sequences for anomaly detection.

##### Caffe: 
The toolbox from [master branch](https://github.com/BVLC/caffe.git) should work. If it doesn't work, please use my modified version [here](https://github.com/t2mhanh/caffe_convLSTM_WTA.git). 
