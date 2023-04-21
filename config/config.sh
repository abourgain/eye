#/bash
# a bash script to setup the environment for the project
# install python and virtualenv
brew install python@3.10
python3.10 -m pip install virtualenv

# create the virtual environment in the project root
python3.10 -m virtualenv eye_env 

# activate the virtual environment
. ./eye_env/bin/activate
echo "export PYTHONPATH=$(pwd)" >> eye_env/bin/activate

# install packages you will need
python3.10 -m pip install -r config/requirements.txt

# install ipykernel
python3.10 -m pip install ipykernel
# create a custom kernel
python3.10 -m ipykernel install --user --name eye_env --display-name "Eye Experiments Kernel"