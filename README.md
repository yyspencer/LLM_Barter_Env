# LLM_Barter_Env

## Environment Setup

This project supports both **Conda** and **pip** workflows.  
Most users should use the **Conda** setup.

### Option 1: Conda (recommended)
Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate barter
```

### Option 2: Pip for non conda users, install the dependencies with: 
```bash
pip install -r requirements.txt
```

Analysis code requires jupyter notebook, register the environment as a notebook kernel:
```bash
python -m ipykernel install --user --name barter --display-name "barter"
```