# to ensure exaact text - conform to convention of squishing text
PREPROCESS_MAIN=mat_preprocessing.py
TEST_LIBRARY=pytest
# Docker configuration
IMAGE_NAME=vt2
CONTAINER_WORKDIR=/app

HOST_DIR=$(PWD)
GPU_FLAG=all
CMD=bash


DISPLAY_VAR=$(DISPLAY) #TODO: evaluat if ) vs none is fine


# Environment variables to pass to container
PROJECT_ROOT=$(CONTAINER_WORKDIR)

DATA_DIR=$(CONTAINER_WORKDIR)/data#/ removed because I can then when I concatenate emphasize prevoius as a directory? another dual interpertation
UNPROCESSED_DATA_DIR=$(CONTAINER_WORKDIR)/data/Matlab/
PROCESSED_DATA_DIR=$(DATA_DIR)/preprocessed_files#what if there are extra slashes? this // is then incorrect?
# DF[0]: ensure no space between the files and the '#' this created an extra space which is bad



MODEL_DIR=$(CONTAINER_WORKDIR)/models


# Added the "f" flag to allow graceful failure, and ignoring messages
clean:
	rm -rf $(PROCESSED_DATA_DIR)
#pp = preprocess
pp: 
	python3 $(PREPROCESS_MAIN) 
# TODO: change and refactor this as needed
openp: 
	vim mat_preprocessing.py
test: 
	$(TEST_LIBRARY)	
# Docker run target
# Added X11 socket volume TODO: refactor this
#
#
docker_run:
	sudo docker run \
		--gpus $(GPU_FLAG) \
		-v $(HOST_DIR):$(CONTAINER_WORKDIR) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		-e PROJECT_ROOT=$(PROJECT_ROOT) \
		-e DATA_DIR=$(DATA_DIR) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e UNPROCESSED_DATA_DIR=$(UNPROCESSED_DATA_DIR)\
		-e PROCESSED_DATA_DIR=$(PROCESSED_DATA_DIR) \
		-w $(CONTAINER_WORKDIR) \
		-it $(IMAGE_NAME) $(CMD)

docker_build:
	sudo docker build -t $(IMAGE_NAME) .

.PHONY: clean build test all docker_run
