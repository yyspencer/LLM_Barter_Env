# LLM Barter Env

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

## API Keys
API keys should not be stored directly in the repository or inside config files.
Instead:
Create a local .env file in the project root, and put your real API keys there.
Make sure .env is ignored by Git through .gitignore.

An example template is provided in .env.example.
Copy it and create your own private .env file with the same variable names. Just change the API keys there. 
## Configs
This project is designed to be configuration-driven. The main config files control the experiment, model assignments, prompts, and evaluation settings.
### models.yaml
Defines:
which providers/models are available
which environment variable contains each provider’s API key
generation settings such as temperature and token limits
how many players in the experiment use each model
### experiment.yaml
Defines the market setup, including:
number of players
number of goods
number of rounds
matching rule
mechanism settings such as sequential vs. simultaneous execution
broadcast / anonymity options
random seed and repeated-run settings

### prompts.yaml

### eval.yaml

## Run code(src)

## Analysis