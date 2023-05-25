#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 32G
#SBATCH --gres gpu:1
#SBATCH --time 1-0:00:00
#SBATCH --job-name experiment-bert-baseline-%J
#SBATCH --output /home/jerabvo1/_logs/experiment-sliding_window--%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@login.rci.cvut.cz

Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: login.rci.cvut.cz
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"


# Load the required modules
ml Python/3.9.5-GCCcore-10.3.0
ml PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
ml PyTorch-Lightning/1.5.9-foss-2021a-CUDA-11.3.1
# ml scikit-learn/0.24.2-foss-2021a
ml torchvision/0.11.1-foss-2021a-CUDA-11.3.1
# ml PyTorch/1.10.0-foss-2021a
# ml IPython/7.13.0-foss-2020a-Python-3.8.2
# ml PyTorch/1.9.0-fosscuda-2020b
# ml scikit-learn/0.24.1-fosscuda-2020b
# ml Python/3.8.2-GCCcore-9.3.0

source ~/my_env/bin/activate

# dataset, window, pma, max are all parameters that we were changing for each experiment
python experiment_e2e_MIL.py --dataset='/home/jerabvo1/_data/multisource-40k-similar/' --run_name='e2e_agg:max_sliding_window_4' --window=4 --pma=0 --max=1

