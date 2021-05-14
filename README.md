# convLSTM_Prediction_AnomalyDetection
We adopt a convolutional Long Short Term Memory (convLSTM)  encoder-decoder to learn normal temporal dynamics from sequences of successive frames. Convolutional LSTMs have already been
employed for anomaly detection. However, the temporal unit is applied on the final spatial stage, which encodes high level representations.
Interleaving RNNs between spatial convolution layers has recently been shown to improve
performance on precipitation now-casting. The model can learn temporal information
on hierarchical spatial representations from low-level to high-level. We adopt the same
architecture. A sequence-to-sequence convolutional LSTM encoder-decoder can be trained for both reconstruction and prediction

In our work, the prediction models outperform the reconstruction models. 



##### Caffe: 
The toolbox from [master branch](https://github.com/BVLC/caffe.git) should work. If it doesn't work, please use my modified version [here](https://github.com/t2mhanh/caffe_convLSTM_WTA.git). 
