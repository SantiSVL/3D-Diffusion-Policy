#!/usr/bin/env bash
# setup_dp3.sh - Entorno DP3 en Onyxia (SSP Cloud), servicio vscode-pytorch-gpu
# Reconstruido a partir de los comandos que funcionaron el 17-nov-2026
# (Ubuntu 24.04, NVIDIA A2, micromamba, Python 3.8, torch 2.0.1+cu118).
# NO se ha probado de una sola pasada: si algo falla, mirad el mensaje del paso.
#
# Uso:
#   export REPO_URL=https://github.com/<usuario>/3D-Diffusion-Policy.git   # solo si el repo no esta clonado
#   bash setup_dp3.sh
#
# Todo lo importante queda en /home/onyxia/work (volumen persistente).

set -e
export WORK=/home/onyxia/work
REPO=$WORK/3D-Diffusion-Policy

step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------
step "1. Librerias del sistema (apt, necesita sudo; se pierden al relanzar el servicio)"
if sudo -n true 2>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y build-essential patchelf libglew-dev libosmesa6-dev \
       libgl1 libglfw3 libx11-dev libegl1 libopengl0 tmux
else
  echo "AVISO: sin sudo. Se omite apt; dependeremos de conda-forge (paso 4)."
fi

# ---------------------------------------------------------------------------
step "2. Repo"
if [ ! -d "$REPO" ]; then
  : "${REPO_URL:?Define REPO_URL con la URL HTTPS de vuestro fork}"
  git clone "${REPO_URL/https:\/\//https://${GIT_PERSONAL_ACCESS_TOKEN}@}" "$REPO"
fi

# ---------------------------------------------------------------------------
step "3. Micromamba y entorno dp3 (Python 3.8)"
mkdir -p $WORK/mm $WORK/.mujoco
if [ ! -x $WORK/bin/micromamba ]; then
  cd $WORK
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$WORK/mm
eval "$($WORK/bin/micromamba shell hook -s bash)"
if [ ! -d $MAMBA_ROOT_PREFIX/envs/dp3 ]; then
  micromamba create -n dp3 python=3.8 -c conda-forge -y
fi

# ---------------------------------------------------------------------------
step "4. MuJoCo 2.1.0"
if [ ! -d $WORK/.mujoco/mujoco210 ]; then
  cd $WORK/.mujoco
  curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar xz
fi
ln -sfn $WORK/.mujoco ~/.mujoco

# ---------------------------------------------------------------------------
step "5. Fichero de activacion ($WORK/activate_dp3.sh)"
cat > $WORK/activate_dp3.sh <<'EOF'
export WORK=/home/onyxia/work
export MAMBA_ROOT_PREFIX=$WORK/mm
eval "$($WORK/bin/micromamba shell hook -s bash)"
micromamba activate dp3
ln -sfn $WORK/.mujoco ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
# variables de compilacion (nvcc, gcc 11 de conda)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
EOF
set +e
source $WORK/activate_dp3.sh
set -e

# ---------------------------------------------------------------------------
step "6. Compiladores y cabeceras desde conda-forge"
micromamba install -c conda-forge -y gcc_linux-64=11 gxx_linux-64=11 ninja
micromamba install -c "nvidia/label/cuda-11.8.0" -y cuda-nvcc cuda-cccl cuda-cudart-dev cuda-libraries-dev
micromamba install -c conda-forge -y xorg-libx11 xorg-xorgproto xorg-libxext xorg-libxrandr \
    xorg-libxinerama xorg-libxcursor xorg-libxi glew mesalib libgl-devel libegl-devel libglu
nvcc --version | tail -2

# ---------------------------------------------------------------------------
step "7. PyTorch 2.0.1 + cu118 y versiones fijadas"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

# ---------------------------------------------------------------------------
step "8. Paquete DP3, mujoco-py y entornos de simulacion"
cd $REPO
# gym 0.21: pip moderno rechaza 'opencv-python>=3.'
sed -i 's/opencv-python>=3\./opencv-python>=3.0/' third_party/gym-0.21.0/setup.py
grep -n opencv third_party/gym-0.21.0/setup.py || true

(cd 3D-Diffusion-Policy && pip install -e .)
(cd third_party/mujoco-py-2.1.2.14 && pip install -e .)
(cd third_party/gym-0.21.0 && pip install -e .)
(cd third_party/Metaworld && pip install -e .)
(cd third_party/rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/.)
# dexart solo hace falta para las tareas DexArt; si falla, no es critico para MetaWorld
(cd third_party/dexart-release && pip install -e .) || echo "AVISO: dexart-release no se instalo (no es critico para MetaWorld)"

# ---------------------------------------------------------------------------
step "9. pytorch3d simplificado (compila CUDA, tarda varios minutos)"
(cd third_party/pytorch3d_simplified && pip install -e .)

# ---------------------------------------------------------------------------
step "10. Paquetes restantes"
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
    einops==0.4.1 diffusers==0.11.1 numba==0.56.4 natsort moviepy imageio av matplotlib termcolor
# diffusers 0.11.1 necesita un huggingface_hub antiguo (cached_download)
pip install "huggingface_hub<0.26"
# visualizador de nubes de puntos (opcional)
pip install kaleido plotly
(cd visualizer && pip install -e .) || echo "AVISO: visualizer no se instalo (opcional)"

# ---------------------------------------------------------------------------
step "11. Pruebas de humo (la primera importacion de mujoco_py compila, 2-5 min)"
export MUJOCO_GL=egl
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import diffusers, zarr, hydra; print('deps OK')"
python -c "import mujoco_py; print('mujoco_py OK')"
python -c "import metaworld, gym; print('metaworld OK')"
python -c "from pytorch3d.ops import sample_farthest_points; print('pytorch3d OK')"
python - <<'PY'
import metaworld, random
ml1 = metaworld.ML1('assembly-v2')
env = ml1.train_classes['assembly-v2']()
env.set_task(random.choice(ml1.train_tasks))
env.reset()
img = env.render(offscreen=True, camera_name='corner', resolution=(84, 84))
print('render OK', img.shape)
PY

echo
echo "Listo. En cada terminal nueva: source $WORK/activate_dp3.sh"
echo "Si falla el render con EGL, probad: export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"