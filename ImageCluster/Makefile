UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
INCLUDE=
LIBS="-L/usr/lib/x86_64-linux-gnu"
endif
ifeq ($(UNAME), Darwin)
INCLUDE="-I/usr/X11R6/include"
LIBS="-L/usr/X11R6/lib"
endif

all: dlib source.o FaceBlendCommon.o ImageCluster.o main.o
	g++ FaceBlendCommon.o ImageCluster.o main.o source.o -std=c++11 -lpthread $(LIBS) -lX11 `pkg-config opencv --libs --cflags` -o ImageCluster

dlib:
	if [ ! -d "dlib-19.4" ]; then \
        unzip -qq "extra/dlib-19.4.zip"; \
		rm -rf __MACOSX; \
	fi

main.o: main.cpp MainHeader.hpp
	g++ -std=c++11 -Idlib-19.4/ -c main.cpp

ImageCluster.o: ImageCluster.cpp ImageCluster.hpp
	g++ -std=c++11 -Idlib-19.4/ -c ImageCluster.cpp

FaceBlendCommon.o: FaceBlendCommon.cpp FaceBlendCommon.hpp
	g++ -std=c++11 -Idlib-19.4/ -c FaceBlendCommon.cpp

source.o: dlib-19.4/dlib/all/source.cpp
	g++ -std=c++11 -Idlib-19.4/ $(INCLUDE) -c dlib-19.4/dlib/all/source.cpp

clean:
	rm -f *.o ImageCluster
