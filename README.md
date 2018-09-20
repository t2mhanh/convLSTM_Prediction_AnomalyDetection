# convLSTM_Prediction_AnomalyDetection
We propose to use encoder-decoders by interleaving convolutional Long Short Term Memory (convLSTM) layers between convolutional layers to learn spatial temporal dynamic in normal video sequence. These encoder-decoders can be reconstruction or prediction. The intuition behind this work is that if we train the encoder-decoder to reconstruct or predict normal sequences with low cost, it should reconstruct/predict abnormal sequences with high cost. 

In our work, the prediction models outperform the reconstruction models. 

##### Caffe: 
The toolbox from [master branch](https://github.com/BVLC/caffe.git) should work. If it doesn't work, please use my modified version [here](https://github.com/t2mhanh/caffe_convLSTM_WTA.git). 
