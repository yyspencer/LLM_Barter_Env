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
