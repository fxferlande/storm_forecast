# CONFIGURATION

# PRODUCTION COMMANDS
main: ## run main project pipeline
	python model/main.py

pipeline: install   ## run main project pipeline
	main

# PROJECT SETUP COMMANDS
install: requirements-dev.txt  ## install project dependencies (requirements-dev.txt)
	pip install -r requirements-dev.txt
	touch install

init: ## initiate virtual environment
	bash init.sh
	touch init
