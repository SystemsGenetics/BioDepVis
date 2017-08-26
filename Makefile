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


all: G3NAV
	rm *.o

%.o: $(SRC)/%.cpp
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

%.o: $(SRC)/%.cu
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

G3NAV: util.o events.o main.o Camera.o Vector.o Utility.o Matrix.o graph.o alignment.o jsoncpp.o cuda_code.o miscgl.o texture.o parse.o lodepng.o Ont.o 
	$(NVCC) -o G3NAV $^ $(LIBS)

clean:
	rm -rf *.o G3NAV
