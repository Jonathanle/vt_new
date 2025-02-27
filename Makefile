# to ensure exaact text - conform to convention of squishing text
OUTPUT_DIRECTORY=cropped_myo_lge_testing
PREPROCESS_MAIN=mat_preprocessing.py
TEST_LIBRARY=pytest
# Docker configuration
IMAGE_NAME=vt2
CONTAINER_WORKDIR=/app
HOST_DIR=$(PWD)
GPU_FLAG=all
CMD=bash
# Environment variables to pass to container
PROJECT_ROOT=$(CONTAINER_WORKDIR)
DATA_DIR=$(CONTAINER_WORKDIR)/data
MODEL_DIR=$(CONTAINER_WORKDIR)/models
# Added the "f" flag to allow graceful failure, and ignoring messages
clean:
	rm -rf $(OUTPUT_DIRECTORY)
#pp = preprocess
pp: 
	python3 $(PREPROCESS_MAIN) 
# TODO: change and refactor this as needed
openp: 
	vim mat_preprocessing.py
test: 
	$(TEST_LIBRARY)	
# Docker run target
docker_run:
	sudo docker run \
		--gpus $(GPU_FLAG) \
		-v $(HOST_DIR):$(CONTAINER_WORKDIR) \
		-e PROJECT_ROOT=$(PROJECT_ROOT) \
		-e DATA_DIR=$(DATA_DIR) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-w $(CONTAINER_WORKDIR) \
		-it $(IMAGE_NAME) $(CMD)
.PHONY: clean build test all docker_run
