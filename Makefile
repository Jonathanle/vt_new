
# to ensure exaact text - conform to convention of squishing text
OUTPUT_DIRECTORY=cropped_myo_lge_testing
PREPROCESS_MAIN=mat_preprocessing.py

clean:
	rm -r $(OUTPUT_DIRECTORY)

#pp = preprocess
pp: 
	python3 $(PREPROCESS_MAIN) 

# TODO: change and refactor this as needed
openp: 
	vim mat_preprocessing.py
.PHONY: clean build test all
