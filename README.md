# LLM Barter Experiment

## Environment Setup

This project supports both **Conda** and **pip** workflows.  
Most users should use the **Conda** setup.

### Option 1: Conda (recommended)
Create and activate the environment
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

## API Keys
API keys should not be stored directly in the repository or inside config files.
Instead:
Create a local .env file in the project root, and put your real API keys there.
Make sure .env is ignored by Git through .gitignore.

An example template is provided in .env.example.
Copy it and create your own private .env file with the same variable names. Just change the API keys there. 

## Run code(src)
