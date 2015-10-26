CC=g++
NVCC=nvcc
SRC = G3NA
INC=$(SRC)
LIB=-lGL -lGLU -lglut

all: main.o Camera.o Vector.o Utility.o Matrix.o graph.o alignment.o jsoncpp.o cuda_code.o
	$(NVCC)  main.o Camera.o Vector.o Utility.o Matrix.o graph.o alignment.o jsoncpp.o cuda_code.o $(LIB) -o G3NAV

main.o:	$(SRC)/main.cpp
	$(NVCC) -c -std=c++11 $(SRC)/main.cpp -I $(INC)

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

clean:
	rm -rf *.o
#/alignment.cpp  G3NA/Camera.cpp  G3NA/graph.cpp  G3NA/jsoncpp.cpp  G3NA/main.cpp  G3NA/Matrix.cpp  G3NA/miscgl.cpp  G3NA/Utility.cpp  G3NA/Vector.cpp


