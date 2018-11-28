INSTALL_PREFIX ?= /usr/local
DEBUG ?= 0

MAKE = make
NVCC = nvcc
NVCCFLAGS = -std=c++11 -Wno-deprecated-gpu-targets

QMAKE = qmake

ifeq ($(DEBUG), 1)
QMAKEFLAGS += "CONFIG+=debug"
endif

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
	cd $(BUILD) && $(QMAKE) .. $(QMAKEFLAGS)
	+$(MAKE) -C $(BUILD)

install: all
	mkdir -p $(INSTALL_PREFIX)/bin
	cp $(BUILD)/BioDepVis $(INSTALL_PREFIX)/bin

clean:
	rm -rf $(BUILD) $(OBJ)
