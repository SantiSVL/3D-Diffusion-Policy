# Installing Conda Environment from Zero to Hero

The following guidance works well for a machine with 3090/A40/A800/A100 GPU, cuda 11.7, driver 515.65.01.

First, git clone this repo and `cd` into it.

    git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git


**Please strictly follow the guidance to avoid any potential errors. Especially, make sure Gym version is the same.**

**Don't worry about the gym version now. Just install my version in `third_party/gym-0.21.0` and you will be fine.**

---
 
## Quick setup on Onyxia / SSP Cloud (fork-specific)
 
This section is specific to our fork. It automates the installation on the SSP Cloud datalab (Onyxia), using the `vscode-pytorch-gpu` service. The environment we validated: Ubuntu 24.04, NVIDIA A2 (15 GB), driver with CUDA 13.4, `sudo` available, micromamba, Python 3.8 and torch 2.0.1+cu118.
 
1. Launch a `vscode-pytorch-gpu` service with 1 GPU and a persistent volume (it is mounted at `/home/onyxia/work`). Clone your fork inside `/home/onyxia/work`.
2. Run the setup script from the root of the repository:
```
export REPO_URL=https://github.com/<usuario>/3D-Diffusion-Policy.git
bash setup_dp3.sh
```
 
`REPO_URL` is only needed if the repository is not cloned yet at `/home/onyxia/work/3D-Diffusion-Policy`. If it is already there, you can simply run `bash setup_dp3.sh`.
 
3. In every new terminal, reactivate the environment with:
```
source /home/onyxia/work/activate_dp3.sh
```
 
What the script does, in order: system libraries with `apt` (if `sudo` is available), micromamba and the `dp3` environment (Python 3.8), MuJoCo 2.1.0, the `activate_dp3.sh` file, gcc 11 / `nvcc` 11.8 / X11 and OpenGL headers from conda-forge, torch cu118, the `opencv-python` fix for `gym-0.21.0`, all the `pip install -e` steps below, `huggingface_hub<0.26` (needed by `diffusers==0.11.1`), and a final smoke test that includes an offscreen render.
 
Notes:
 
- Everything important lives in `/home/onyxia/work`, which is the persistent volume. Packages installed with `apt` are lost when the service is relaunched, so rerun the script (it skips what already exists) or at least step 1.
- If EGL rendering fails, use `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (CPU rendering, slower).
- If `import mujoco_py` fails after cleaning a build, do not delete the whole `third_party/mujoco-py-2.1.2.14/mujoco_py/generated` folder: it contains `wrappers.pxi`. Remove only `generated/_pyxbld_*`, `cymj.c` and `cymj*.so`.
- `dexart-release` is installed with a warning instead of an error, because it is only needed for the DexArt tasks and not for MetaWorld.
- The script was reconstructed from the commands that worked step by step, not tested in a single clean run. If a step fails, the `=== N. ... ===` header tells you which one.
The manual steps below are the original instructions, kept for reference. The script above replaces them.
 

---

1.create python/pytorch env

    conda remove -n dp3 --all
    conda create -n dp3 python=3.8
    conda activate dp3


---

2.install torch

    # if using cuda>=12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version

---

3.install dp3

    cd 3D-Diffusion-Policy && pip install -e . && cd ..


---

4.install mujoco in `~/.mujoco`

    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz

and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl


and then install mujoco-py (in the folder of `third_party`):

    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..


----

5.install sim env

    pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

    cd third_party
    cd dexart-release && pip install -e . && cd ..
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..

download assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing), unzip it, and put it in `third_party/dexart-release/assets`. 

download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.

---

6.install pytorch3d (a simplified version)

    cd third_party/pytorch3d_simplified && pip install -e . && cd ..


---

7.install some necessary packages

    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor


---

8.install our visualizer for pointclouds (optional)

    pip install kaleido plotly
    cd visualizer && pip install -e . && cd ..

