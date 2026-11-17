export WORK=/home/onyxia/work
export MAMBA_ROOT_PREFIX=$WORK/mm
eval "$($WORK/bin/micromamba shell hook -s bash)"
micromamba activate dp3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib
export TORCH_CUDA_ARCH_LIST="8.6"
ln -sfn $WORK/.mujoco ~/.mujoco
