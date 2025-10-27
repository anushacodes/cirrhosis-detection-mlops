# Makefile

# install dependencies with pip or uv
install-pip:
	pip install -r requirements.txt	

install-uv:
	uv pip install -r requirements.txt	


format:
	black *.py src/*.py

lint:
	flake8 *.py src/*.py

dvc:
	dvc init
	dvc repro

test:
	uv run test.py
	
init: install-uv format dvc
	

build:
	docker build -t cirrhosis-detector .
	docker run -p $(APP_PORT):$(APP_PORT) cirrhosis-detector

serve:
	uvicorn app.main:app --reload --port $(APP_PORT)


clean:
	rm -rf __pycache__ .pytest_cache .dvc/tmp
	find . -type f -name "*.pyc" -delete
