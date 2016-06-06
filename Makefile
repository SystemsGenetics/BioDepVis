CC=g++
NVCC=nvcc
SRC = G3NA
INC=-I ./glui-2.35/src/include 
GLUILIB=-L ./glui-2.35/src/lib
#GLEWLIB=-L /home/benafsh/NVIDIA_CUDA-7.5_Samples/common/lib/linux/x86_64
PLATFORM= $(shell uname -s)

ifeq "$(PLATFORM)" "Darwin"
	LIB=-Xlinker -framework,GLUT -Xlinker -framework,OpenGL 
endif

ifeq "$(PLATFORM)" "Linux"
	LIB=-lGL -lGLU -lglut $(GLUILIB) -lglui
endif


all: main.o Camera.o Vector.o Utility.o Matrix.o graph.o alignment.o jsoncpp.o cuda_code.o miscgl.o texture.o parse.o lodepng.o Ont.o
	$(NVCC)  main.o Camera.o Vector.o Utility.o Matrix.o graph.o alignment.o jsoncpp.o cuda_code.o miscgl.o texture.o parse.o lodepng.o Ont.o $(LIB) -o G3NAV.exe
	rm *.o

main.o:	$(SRC)/main.cpp
	$(NVCC) -c -std=c++11 $(SRC)/main.cpp $(INC)

Camera.o: $(SRC)/Camera.cpp
	$(NVCC) -c $(SRC)/Camera.cpp

Vector.o: $(SRC)/Vector.cpp
	$(NVCC) -c $(SRC)/Vector.cpp

Utility.o: $(SRC)/Utility.cpp
	$(NVCC) -c $(SRC)/Utility.cpp

Matrix.o: $(SRC)/Matrix.cpp
	$(NVCC) -c $(SRC)/Matrix.cpp

graph.o: $(SRC)/graph.cpp
	$(NVCC) -c -std=c++11 $(SRC)/graph.cpp

alignment.o: $(SRC)/alignment.cpp
	$(NVCC) -c -std=c++11 $(SRC)/alignment.cpp

jsoncpp.o: $(SRC)/jsoncpp.cpp
	$(NVCC) -c -std=c++11 $(SRC)/jsoncpp.cpp

cuda_code.o: $(SRC)/cuda_code.cu
	$(NVCC) -c -std=c++11 $(SRC)/cuda_code.cu

miscgl.o: $(SRC)/miscgl.cpp
	$(NVCC) -c -std=c++11 $(SRC)/miscgl.cpp

texture.o: $(SRC)/texture.cpp
	$(NVCC) -c -std=c++11 $(SRC)/texture.cpp

parse.o: $(SRC)/parse.cpp
	$(NVCC) -c -std=c++11 $(SRC)/parse.cpp

lodepng.o : $(SRC)/lodepng.cpp
	$(NVCC) -c -std=c++11 $(SRC)/lodepng.cpp

Ont.o : $(SRC)/Ont.cpp
	$(NVCC) -c -std=c++11 $(SRC)/Ont.cpp
clean:
	rm -rf *.o
	rm -rf G3NAV.exe
#/alignment.cpp  G3NA/Camera.cpp  G3NA/graph.cpp  G3NA/jsoncpp.cpp  G3NA/main.cpp  G3NA/Matrix.cpp  G3NA/miscgl.cpp  G3NA/Utility.cpp  G3NA/Vector.cpp


