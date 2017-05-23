#!/bin/sh
# Script to install libglui.

wget https://github.com/libglui/glui/archive/2.37.zip -O glui-2.37.zip
unzip glui-2.37.zip

cd glui-2.37 && make && cd ..

rm -rf glui-2.37.zip
