# CONFIGURATION

# PRODUCTION COMMANDS
main: ## run main project pipeline
ifdef backend
	export KERAS_BACKEND='$(backend)'; python model/main.py
else
	@echo 'No backend variable passed, using Tensorflow'
	export KERAS_BACKEND='tensorflow'; python model/main.py
endif

pipeline: install   ## run main project pipeline
	main

# PROJECT SETUP COMMANDS
install: ## install project dependencies (requirements-dev.txt)
	pip install -r requirements-dev.txt
	touch install

init: ## initiate virtual environment
	bash init.sh
	touch init
