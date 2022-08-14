#!/bin/bash
[ -e libwarping.so ] && rm libwarping.so
gcc -fPIC -shared -o libwarping.so warping.cpp
