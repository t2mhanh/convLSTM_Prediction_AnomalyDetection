# Example script: sample_script.sh
#!/bin/bash
# Set current working directory
#$ -cwd
#Use current environment variables/ modules
#$ -V
#$ -l coproc_p100=1
#Request one hour of runtime
#$ -l h_rt=48:00:00
#$ -l disk=1G
#Email at the beginning and end of the job
#$ -m be
#Run the executable 'myprogram' from the current working directory
#./hello.sh

module add cuda
module add singularity
singularity exec --nv -B $TMPDIR:/tmp -B /nobackup/schtmt /nobackup/schtmt/containers/caffe_lstm_wta_gpu.img /opt/caffe/build/tools/caffe train --solver=/nobackup/schtmt/home_temp/ucsd_conv_lstm_frame_prediction_6/solver_ped1_augmentation_2.prototxt --weights=/nobackup/schtmt/conv_wta_lstm_model/convLSTM_prediction6_aug_ped1_old/_iter_1657500.caffemodel 2>&1|tee /nobackup/schtmt/home_temp/ucsd_conv_lstm_frame_prediction_6/ped1_augmentation_old_2.log


