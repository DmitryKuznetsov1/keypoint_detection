DATASETS_PATH ?= tasks
BATCH_SIZE ?= 2
LOAD_IMAGES ?= 0

create-venv:
	python3 -m venv venv

activate-venv:
	. venv/bin/activate

load-model:
	python build/model_preloading.py

install-deps:
	pip install -r requirements.txt

evaluate_model:
	python main.py --datasets-path $(DATASETS_PATH) --batch-size $(BATCH_SIZE) --load-to-memory $(LOAD_IMAGES)

setup: create-venv activate-venv install-deps load-model

eval: activate-venv evaluate_model
