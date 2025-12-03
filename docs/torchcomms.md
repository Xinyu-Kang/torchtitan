# TorchTitan + TorchComms on MI325X Cluster
## Create Conda Environment
```
conda create -n comms-titan python=3.12
conda activate comms-titan
```
Install PyTorch:
```
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

## Build TorchComms from Source
### Clone Repo Fork:
```
https://github.com/Xinyu-Kang/torchcomms.git
cd torchcomms
```
### Build RCCLX Backend
Install some prerequisites:
```
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y
```
Environment variables to find rocm/rcclx headers:
```
export BUILD_DIR=${PWD}/comms/rcclx/develop/build/release/build
export ROCM_HOME=/opt/rocm
export RCCLX_INCLUDE=${BUILD_DIR}/include/rccl
export RCCLX_LIB=${BUILD_DIR}/lib
```
Run build script:
```
./build_rcclx.sh --amdgpu_targets gfx942
```
### Install TorchComms
Set backend env vars before installing:
```
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=OFF
export USE_RCCL=OFF
export USE_RCCLX=ON
```
Install requirements:
```
pip install -r requirements.txt
```
Install TorchComms:
```
pip install --no-build-isolation -v .
```
## Build TorchTitan from Source
### Clone Repo Fork
```
cd ..
git clone https://github.com/Xinyu-Kang/torchtitan.git
cd torchtitan
```
### Install TorchTitan
Install requirements:
```
pip install -r requirements.txt
```
Install TorchTitan:
```
pip install -e .
```
## Run TorchTitan with TorchComms
### Download Model
Llama-3.1-8B:
```
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token="$HF_TOKEN"
```
Llama-3.1-70B:
```
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-70B --assets tokenizer --hf_token="$HF_TOKEN"
```
### Download Dataset
```
python scripts/download_c4_test_dataset.py --save_dir ./assets/hf/c4_test
```
### Single-Node Training
```
TEST_BACKEND=ncclx TRAIN_FILE=torchtitan.experiments.torchcomms.train CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```
### Multi-Node Training (SLURM)
With TorchComms enabled:
```
sbatch ./run_slurm_multinode.sh
```
Without TorchComms:
```
USE_TORCHCOMMS=0 sbatch ./run_slurm_multinode.sh
```
To change the model to train, edit `CONFIG_FILE` inside `run_slurm_multinode.sh`; the default config file is `./torchtitan/models/llama3/train_configs/llama3_8b.toml` 
