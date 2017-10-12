NVCC = nvcc
NVCCFLAGS = -std=c++11 -Wno-deprecated-gpu-targets

BUILD = build
OBJ = obj
SRC = src
BINS = BioDepVis

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(OBJ):
	mkdir -p $(OBJ)

$(OBJ)/fdl_cuda.o: $(SRC)/fdl.cu | $(OBJ)
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

BioDepVis: $(OBJ)/fdl_cuda.o | $(BUILD)
	cd $(BUILD) && qmake .. && make && cp $@ ../

clean:
	rm -rf $(BUILD) $(OBJ) $(BINS)
