# Installing GPU drivers:
## https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py &&
sudo python3 install_gpu_driver.py &&
sudo apt -y install python3-pip && 
sudo apt-get install zlib1g-dev &&
sudo apt-get install -y libjpeg-dev && 
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html &&
pip3 install pytorch-forecasting==0.9.2 pytorch-lightning==1.5.4 &&
sudo apt install git-all



# Intel(R) optimized Base (with Intel(R) MKL and CUDA 11.0)

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html &&
pip install pytorch-forecasting==0.9.2 install pytorch-lightning==1.5.4

git clone https://github.com/fornasari12/temporal-fusion-transformer.git

export NCCL_P2P_DISABLE=1 && export NCCL_IB_DISABLE=1 && export NCCL_DEBUG=WARN && python train_tft.py