#-------------------------------------------------
#
# Project created by QtCreator 2017-08-30T16:48:24
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

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


SOURCES += \
    alignment.cpp \
    database.cpp \
    graph.cpp \
    main.cpp \
    mainwindow.cpp \
    visualizer.cpp

HEADERS += \
    alignment.h \
    database.h \
    graph.h \
    mainwindow.h \
    visualizer.h

FORMS += \
    mainwindow.ui
