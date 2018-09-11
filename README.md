# convLSTM_Prediction_AnomalyDetection
We combine reconstruction/prediction over many frames with interleaved LSTMs to train an end-to-end deep model 
for learning normal appearances and motions in video sequences. Then deriving an anomaly score from 
reconstruction/prediction error for anomaly detection.

##### Caffe: 
The toolbox from [master branch](https://github.com/BVLC/caffe.git) should work. If it doesn't work, please use my modified version [here](https://github.com/t2mhanh/caffe_convLSTM_WTA.git). 
