# Example script: sample_script.sh
#!/bin/bash
# Set current working directory
#$ -cwd
#Use current environment variables/ modules
#$ -V
#$ -l coproc_k80=1
#Request one hour of runtime
#$ -l h_rt=48:00:00
#Email at the beginning and end of the job
#$ -m be
#Run the executable 'myprogram' from the current working directory
#./hello.sh

module add cuda
module add singularity
singularity exec --nv -B /nobackup/schtmt /nobackup/schtmt/containers/caffe_lstm_wta_gpu.img python error_train_avenue.py


