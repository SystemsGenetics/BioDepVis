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
    qt/alignment.cpp \
    qt/database.cpp \
    qt/fdl.cpp \
    qt/glalignobject.cpp \
    qt/glboxobject.cpp \
    qt/glgraphobject.cpp \
    qt/glwidget.cpp \
    qt/graph.cpp \
    qt/main.cpp \
    qt/mainwindow.cpp \
    qt/matrix.cpp

HEADERS += \
    qt/alignment.h \
    qt/database.h \
    qt/fdl.h \
    qt/glalignobject.h \
    qt/glboxobject.h \
    qt/glgraphobject.h \
    qt/globject.h \
    qt/glwidget.h \
    qt/mainwindow.h \
    qt/graph.h \
    qt/matrix.h \
    qt/vector.h
