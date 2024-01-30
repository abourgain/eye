### ---------------   Install all dependencies for the project  --------------- ###
setup:
	. ./config/config.sh

requirements:
	pip install -r config/requirements.txt

### ---------------   Lint  --------------- ###
pylint-src:
	pylint --rcfile=pylint.conf src

lint:
	make pylint-src

### ---------------   Format  --------------- ###
format:
	isort --profile black -l 200 src/
	black -l 200 src/