CC=emcc
SOURCEFILES=interface.cpp OpenNL_psm/OpenNL_psm.cpp
OUTPUT_FOLDER=./build
OUTPUT=$(OUTPUT_FOLDER)/MeshLib.js
# SRC = $(wildcard *.cpp)

compile: .clear
	$(CC) $(SOURCEFILES) -o $(OUTPUT) -O3 --std=c++17 --bind
	cp build/* ../Front/build/

.clear:
	rm -f $(OUTPUT_FOLDER)/*

