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

OBJDIR = obj
OBJS = \
	$(OBJDIR)/alignment.o \
	$(OBJDIR)/Camera.o \
	$(OBJDIR)/cuda_code.o \
	$(OBJDIR)/events.o \
	$(OBJDIR)/graph.o \
	$(OBJDIR)/jsoncpp.o \
	$(OBJDIR)/lodepng.o \
	$(OBJDIR)/main.o \
	$(OBJDIR)/Matrix.o \
	$(OBJDIR)/miscgl.o \
	$(OBJDIR)/Ont.o \
	$(OBJDIR)/parse.o \
	$(OBJDIR)/texture.o \
	$(OBJDIR)/util.o \
	$(OBJDIR)/Utility.o \
	$(OBJDIR)/Vector.o
BINS = biodep-vis

all: $(BINS)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRC)/%.cpp | $(OBJDIR)
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

$(OBJDIR)/%.o: $(SRC)/%.cu | $(OBJDIR)
	$(NVCC) -c $(GLUIINC) -std=c++11 -o $@ $<

biodep-vis: $(OBJS)
	$(NVCC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(OBJDIR) $(BINS)
