# idr_accelerate
## WORK IN PROGRESS !!!


## Description

Make Accelerate on Jean Zay easy.

## Usage
idr_accelerate.py is a script to create the right config files to use Accelerate with several nodes on Jean Zay.
It should be run once on the main process node first to create the config files, then eache node will run (thanks to srun) the python script with the right config file (with the command srun bash -c '...').

Follow the example to have a better understanding.
