CXX  = g++
NVCC = nvcc -Wno-deprecated-gpu-targets

GLUIDIR = glui-2.37
SRC     = src

GLUIINC = -I $(GLUIDIR)/include
GLUILIB = -L $(GLUIDIR)/lib

PLATFORM = $(shell uname -s)

ifeq "$(PLATFORM)" "Darwin"
	LIBS = -Xlinker -framework,GLUT -Xlinker -framework,OpenGL
endif

ifeq "$(PLATFORM)" "Linux"
	LIBS = -lGL -lGLU -lglut $(GLUILIB) -lglui
endif

BINS = biodep-vis
OBJS = \
	alignment.o \
	Camera.o \
	cuda_code.o \
	events.o \
	graph.o \
	jsoncpp.o \
	lodepng.o \
	main.o \
	Matrix.o \
	miscgl.o \
	Ont.o \
	parse.o \
	texture.o \
	util.o \
	Utility.o \
	Vector.o

all: $(BINS)
	rm *.o

%.o: $(SRC)/%.cpp
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

%.o: $(SRC)/%.cu
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

biodep-vis: $(OBJS)
	$(NVCC) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o $(BINS)
