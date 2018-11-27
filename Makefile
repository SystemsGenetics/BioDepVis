INSTALL_PREFIX ?= /usr/local

MAKE = make
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

BioDepVis: $(OBJ)/fdl_cuda.o $(SRC)/*.h $(SRC)/*.cpp  | $(BUILD)
	cd $(BUILD) && qmake ..
	+$(MAKE) -C $(BUILD)

install: all
	mkdir -p $(INSTALL_PREFIX)/bin
	cp $(BUILD)/BioDepVis $(INSTALL_PREFIX)/bin

clean:
	rm -rf $(BUILD) $(OBJ)
