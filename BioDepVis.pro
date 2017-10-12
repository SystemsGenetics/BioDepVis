QT += widgets
TARGET = BioDepVis
TEMPLATE = app
CONFIG += c++11 debug

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000

LIBS += \
    -L/usr/local/cuda/lib64 -lcudart

OBJECTS += \
    obj/fdl_cuda.o

SOURCES += \
    src/alignment.cpp \
    src/database.cpp \
    src/fdl.cpp \
    src/glalignobject.cpp \
    src/glboxobject.cpp \
    src/glgraphobject.cpp \
    src/glwidget.cpp \
    src/graph.cpp \
    src/main.cpp \
    src/mainwindow.cpp \
    src/matrix.cpp

HEADERS += \
    src/alignment.h \
    src/database.h \
    src/fdl.h \
    src/glalignobject.h \
    src/glboxobject.h \
    src/glgraphobject.h \
    src/globject.h \
    src/glwidget.h \
    src/mainwindow.h \
    src/graph.h \
    src/matrix.h \
    src/vector.h
