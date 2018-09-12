USE_VOT_TRAX=0

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm
CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
OBJS =

ifeq ($(USE_VOT_TRAX), 1)
CXXFLAGS+= -ltrax
LDFLAGS+= -ltrax 
endif

ALL = runopencvtrackers.bin
ifeq ($(USE_VOT_TRAX), 1)
	ALL+= vot_opencv.bin 
endif

all: $(ALL)

runopencvtrackers.bin: $(OBJS) runopencvtrackers.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

ifeq ($(USE_VOT_TRAX), 1)
vot_opencv.bin: $(OBJS) vot_opencv.o 
	$(CC) -o $@ $^ $(LDFLAGS) 
endif

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


.PHONY: clean

clean:
	rm -rf *.o *.bin

