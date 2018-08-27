
USE_MULTI_THREAD=1

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm -lopentracker
# -O3 double the speed!!!!!!!!!
CXXFLAGS = -g -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x -O3 -fPIC
HEADERS = eco.hpp

ifeq ($(USE_MULTI_THREAD), 1)
CXXFLAGS+= -DUSE_MULTI_THREAD
LDFLAGS+= -pthread
endif

all: runecotracker.bin

runecotracker.bin: runecotracker.o
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cc $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	rm -rf ./.d *.o *.bin *.so *.a */*.o */*.bin
