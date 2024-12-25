# Stable-V2A - Basic instructions



## Usage

Install the requirements (it is recommended to use Python version 3.8.10)

You can do this with

```
conda env create -f environment.yml
```

or with

```
conda env create --name stablefoley python=3.8.10
conda activate stablefoley
pip install -r requirements.txt
```

We didn't test recently the creation of a new container from scratch, so there might be errors. **Please report any further step you did to make it work, so that we can integrate any missing dependencies.**

### Pip installation on Windows:

For creating a virtual environment on windows and install dependencies using Pip (not conda), copy-and-paste from [this link](https://github.com/YusuphaJuwara/RL_2023/blob/main/virtual_env_setup.md)

Full example:

1. Create a virtual environment

```
C:\Users\yusup\AppData\Local\Programs\Python\Python38-32\python.exe -m venv stablefoley
```

2. Activate the venv

```
stablefoley\Scripts\activate
```

3. Install dependencies

pip install -r requirements.txt
```

## Checkpoints

You can download Stable-Foley, CLAP and CAVP checkpoints through this [Google Drive](https://drive.google.com/drive/folders/1A4b1fKQyIy8h9EOmQGU_8WxLL7Fvfd6j) folder. 
You should put all weights in the `\logs` folder. The result should stick to the following file structure:

```
logs/
    cavp_ckpt/
        # CAVP ckpt
    ckpts/
        # Stable-Foley ckpt
    clap_ckpt/
        # CLAP ckpt
```

Remember to log in on Hugginface with `huggingface-cli login` (this can be done in the [Inference Notebook](/notebook/inference.ipynb)) using personal token in order to be able to download Stable Audio Open weights.

## Dataset

#### Greatest Hits
Download the ZIP archive containing all the files of the dataset. Place it somewhere and adjust the paths in the config files accordingly.

## Training

After changing all paths in the config files and set the wandb entity and api key, you can run the `train.py` script as follows:

```
PYTHONUNBUFFERED=1 TAG=gh-controlnet python3 train.py exp=train_gh_controlnet.yaml
```

When you'll get access to the server rember to:
- Launch the script on the GPU that was previously assigned to you by prepending `CUDA_VISIBLE_DEVICES=ID`, where `ID` is in the set ${0, ..., 4}$.
- Start a Tmux session so that the training can keep running even after closing your terminal. As a preliminary action, use `tmux new -s NAME_SESSION` to start a new tmux session (or `tmux attach -t NAME_SESSION` to restore an existing one). [Here](https://tmuxcheatsheet.com/) you can see the full set of instructions.

## Inference

Use the inference notebook in the `notebook` folder.