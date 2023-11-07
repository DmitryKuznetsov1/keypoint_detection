DATASETS_PATH ?= tasks
BATCH_SIZE ?= 2
LOAD_IMAGES ?= 0

setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt
	./venv/bin/python3 build/model_preloading.py

eval:
	./venv/bin/python3 main.py --datasets-path $(DATASETS_PATH) --batch-size $(BATCH_SIZE) --load-to-memory $(LOAD_IMAGES)
