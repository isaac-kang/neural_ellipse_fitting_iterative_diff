ll
python
nvidia-smi
ll
mkdir project
ll
cd project
ll
nvidia-smi
exit
ll
cd project/
ll
nvidia-smi
ll
nvidia-smi
ll
cd hcnet_org_v1_multigpu/
ll
rm *.avi
ll
cd code_training/
ll
cat hgnet.py 
exit
ll
nvidia-smi
ll
exit
nvidia-smi
ll
nvidia-smi
ll
nvidia-smi
ll
cd project/
ll
cd hcnet_org_v1_multigpu/
ll
python ./code_training/main.py
conda create --name py36 python=3.6
pip install conda
conda create --name py36 python=3.6
conda info
anaconda
./source
exit
conda
conda create 
ll
cd project/
ll
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh 
conda 
conda -V
ll
source
exit
conda
pip uninstall conda
conda
ll
cd project/
ll
./Miniconda3-latest-Linux-x86_64.sh 
conda
exit
conda
./Miniconda3-latest-Linux-x86_64.sh -u
cd project/
ll
./Miniconda3-latest-Linux-x86_64.sh -u
conda
exit
ll
cd project/
ll
rm Miniconda3-latest-Linux-x86_64.sh 
cd hcnet_org_v1_multigpu/
ll
nvidia-smi
ll
source activate py36
ll
cd experiments/
ll
cd 1st_test/
ll
tensorboard --logdir=./
ll
nvidia-smi
tensorboard --logdir=./
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
conda
conda create --name py36 python=3.6
pip install tensorflow-gpu
ll
source activate py36
ll
cd project/
ll
cd hcnet_org_v1_multigpu/
python ./code_training/main.py 
pip install numpy
python ./code_training/main.py 
ll
pip install tensorflow-gpu
python ./code_training/main.py 
pip install opencv-python
python ./code_training/main.py 
pip install dlib
python ./code_training/main.py 
pip install skimage
pip install scikit-image
python ./code_training/main.py 
pip install tqdm
python ./code_training/main.py 
pip install sharedmem
python ./code_training/main.py 
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py 
ll
cd project/
ll
cd hcnet_org_v1_
cd hcnet_org_v1_512x512/
ll
cd experiments/
ll
nvidia-smi
ll
cd 512x512/
ll
TRAINING_FILE_PATHS = [  ]
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
cd ..
ll
cd 512x512_3stack_fusion_at_the_last/
ll
nvidia-smi
ll
rm *
cd ..
ll
rm 512x512_3stack_fusion_at_the_last/
rmdir 512x512_3stack_fusion_at_the_last/
cd ..
ll
tensorboard --logdir=./
ll
exit
nvidia-smi
htop
ps -a
source activate py36
killall python
ll
cd project/
ll
exit
ll
cd project/
ll
cd hcnet_org_v1_512x512/
ll
source activate py36
./code_training/main.py
CUDA_VISIBLE_DEVICES=2,3 ./code_training/main.py
CUDA_VISIBLE_DEVICES=2,3 python ./code_training/main.py
ll
rm *.avi
ll
./code_training/main.py
python ./code_training/main.py
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
ll
ssh hikoo@192.168.86.44
ll
CUDA_VISIBLE_DEVICES=0 python ./code_training/main.py
exit
cd project
ll
cd hcnet_org_v1_variable_inputsize_and_stack/
ll
cd experiments/
ll
rmdir 512x512/
rmdir 256x256_256_3_8/
cd 256x256_256_3_8
ll
rm *
cd ..
rmdir 256x256_256_3_8
ll
cd 256x256_256_3_16/
ll
source activate py36
tensorboard --logdir=./
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
exit
ll
cd project/
ll
cd hcnet_org_v1_variable_inputsize_and_stack/
ll
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
source activate py36
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
exit
ll
cd project/
ll
cd hcnet_org_v1_appearance_focus/
ll
source activate py36
nvidia-smi
CUDA_VISIBLE_DEVICES=2,3 source activate py36
CUDA_VISIBLE_DEVICES=2,3 python ./code_training/main.py
cat ./code_commons/global_constants.py 
CUDA_VISIBLE_DEVICES=2,3 python ./code_training/main.py
exit
cd project/
ll
cd hcnet_org_v1_appearance_focus/
ll
cd experiments/
ll
cd 256x256_512_2_8/
ll
rm *
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm events.out.tfevents.1558107659.lambda-quad 
source activate py36
tongue_wght = 0
eyeball_wght = 0
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm *
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
nvidia-smi
exit
nvidia-smi
source activate py36
ll
cd project/
ll
cd hcnet_org_v1_variable_inputsize_and_stack/
ll
./code_training/main.py
python ./code_training/main.py
CUDA_VISIBLE_DEVICES=2,3 python ./code_training/main.py
cd ..
ll
cd hcnet
python ./code_training/main.py --batch_size 16  --NUM_STACKS 3 --mode train_heatmap --data_dir  "../data/v4_20181112_face111/trainset,../data/v4_20181112_face111/trainset_emilia" --regression_model "./experiments/pretrained/myckpt-317000" --experiment_name lamdba_training  --validation_dataset_file_path "../data/v4_20181112_face111/v4_20181112_dataset_testset.tfrecords"
nvidia-smi
CUDA_VISIBLE_DEVICES=2,3 python ./code_training/main.py --batch_size 16  --NUM_STACKS 3 --mode train_heatmap --data_dir  "../data/v4_20181112_face111/trainset,../data/v4_20181112_face111/trainset_emilia" --regression_model "./experiments/pretrained/myckpt-317000" --experiment_name lamdba_training  --validation_dataset_file_path "../data/v4_20181112_face111/v4_20181112_dataset_testset.tfrecords"
exit
cd project/
source activate py35
source activate py36
ll
cd hcnet_org_v1_variable_inputsize_and_stack/
ll
cd experiments/
ll
cd regression_only_with_autoannotation_256_2_16/
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm events.out.tfevents.1558396846.lambda-quad 
ll
cd ..
ll
cd hcnet
ll
cd experiments/
ll
cd lamdba_training_256_3_16/
ll
cd v..
cd ..
ll
rmdir lamdba_training_256_3_16/
exit
cd project/
ll
cd hcnet_doublefusion/
nvidia-smi
ll
cd experiments/
ll
rmdir lambda_training_256_3_16/
ll
cd lambda_training_256_3_16/
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
nvidia-smi
exit
cd project/
ll
cd hcnet_doublefusion/
ll
source activate py36
CUDA_VISIBLE_DEVICES=2,3  python ./code_training/main.py --batch_size 16  --NUM_STACKS 3 --mode train_heatmap --data_dir  "../data/v4_20181112_face111/trainset,../data/v4_20181112_face111/trainset_emilia" --regression_model "./experiments/pretrained/myckpt-317000" --experiment_name lambda_training  --validation_dataset_file_path "../data/v4_20181112_face111/v4_validation_2506.tfrecords"
exit
ll
cd project/
ll
cd hcnet_org_v1_appearance_focus/
ll
source actviate py36
cd experiments/
ll
cd 256x256_512_4_4/
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm events.out.tfevents.15583*
ll
rm myckpt-5*
ll
rm myckpt-3*
rm myckpt-4*
ll
cd ..
cd hcnet_org_v1_variable_inputsize_and_stack_train_heatmap/
cd experiments/
ll
cd regression_only_with_autoannotation_256_2_16/
ll
rm events.out.tfevents.1558470852.lambda-quad 
cat checkpoint 
ll
cd ..
ll
cd regression_only_with_autoannotation_256_2_16/
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
cat checkpoint
ll
rm events.out.tfevents.1558471027.lambda-quad 
ll
rm events.out.tfevents.1558471778.lambda-quad 
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm events.out.tfevents.1558471973.lambda-quad 
cd ..
ll
cd ..
cd hcnet
ll
cd experiments/
ll
cd lambda_training_256_3_16/
ll
source activate py36
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
ll
rm events.out.tfevents.1558628755.lambda-quad 
ll
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./
exit
ll
nvidia-smi
ll
source activate py36
ll
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
cd project/
ll
cd hcnet_org_v1_appearance_focus/
ll
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
ll
cd ..
ll
cd hcnet_org_v1_variable_inputsize_and_stack_train_heatmap/
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
ll
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
ll
CUDA_VISIBLE_DEVICES=0,1 python ./code_training/main.py
cd ..
cd hcnet
ll
CUDA_VISIBLE_DEVICES=0,1  python ./code_training/main.py --batch_size 16  --NUM_STACKS 3 --mode train_heatmap --data_dir  "../data/v4_20181112_face111/trainset,../data/v4_20181112_face111/trainset_emilia" --regression_model "./experiments/pretrained/myckpt-317000" --experiment_name lambda_training  --validation_dataset_file_path "../data/v4_20181112_face111/v4_20181112_dataset_testset.tfrecords"
CUDA_VISIBLE_DEVICES=0,1  python ./code_training/main.py --batch_size 16  --NUM_STACKS 3 --mode train_heatmap --data_dir  "../data/v4_20181112_face111/trainset,../data/v4_20181112_face111/trainset_emilia" --regression_model "./experiments/pretrained/myckpt-317000" --experiment_name lambda_training  --validation_dataset_file_path "../data/v4_20181112_face111/v4_validation_2506.tfrecords"
exit
ll
nvidia-smi
ll
exit
ll
nvidia-smi
exit
ll
source activate py36
ll
nvidia-smi
cd project/
ll
cd hcnet
cd hcnet_159
cd..
cd ..
cd hcnet_
cd hcnet_159/
ll
cd experiments/
ll
cd lambda_training_256_2_159/
ll
cd ..
ll
cd ..
ll
ps -a
ps -A
ps
ps -a
ps -ef
ps -ef -a
ps -a
ps -ef | grep hikoo
kill 2295
ps -ef | grep hikoo
nvidia-smi
python ./code_training/main.py --batch_size 16 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
CUDA_VISIBLE_DEVICES=1, python ./code_training/main.py --batch_size 16 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 16 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
ll
nvidia-smi
ps -a
ps -ef | grep hiko
ps -ef | grep hikoo
kill 2777
kill 2785
ps -ef | grep hikoo
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 16 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
ll
tree -d
apt install tree
cd experiments/
ll
cd ..
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 8 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training_batch_8 --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
ll
ps -ef | grep hikoo
source activate py36
ll
cd project/
ll
cd hcnet_159/
ll
ps -ef | grep hikoo
ps -a
ps -ef 
nvidia-smi
ll
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 8 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training_batch_8 --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 24 --NUM_STACKS 2 --mode train_regressor --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training_batch_24 --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
ll
cd experiments/
ll
cd lambda_training_
ll
cd lambda_training_
cd lambda_training_batch_24_256_2_159/
ll
mkdir backpu
ll
cp * ./backpu/
cp myckpt-13* ./backpu/
ll
cd backpu/
ll
cd ..
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 24 --NUM_STACKS 2 --mode train_heatmap --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training_batch_24 --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
cd ..
CUDA_VISIBLE_DEVICES=1 python ./code_training/main.py --batch_size 24 --NUM_STACKS 2 --mode train_heatmap --data_dir ../data/v5_20190517_face159/image_bucket --experiment_name lambda_training_batch_24 --validation_dataset_file_path ../data/v5_20190517_face159/v5_validation.tfrecords
