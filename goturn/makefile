
CC=gcc

VPATH=./helper:./loader:./network:./tracker:/media/elab/sdd/caffe/build/install

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/caffe/build/install/lib -lcaffe-nv -lproto \
	-lboost_system -lboost_filesystem -lboost_regex -lglog

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x -I/media/elab/sdd/caffe/build/install/include/ \
	-I/usr/local/cuda/include/

DEPS = $(wildcard */*.h) $(wildcard /media/elab/sdd/caffe/build/install/include/*/*.hpp)

#$(info DEPS is: $(DEPS))

OBJS = helper/bounding_box.o helper/helper.o helper/high_res_timer.o helper/image_proc.o \
		loader/loader_base.o network/regressor_base.o network/regressor.o \
		tracker/tracker.o rungoturntracker.o

all: rungoturntracker.bin classification.bin

rungoturntracker.bin: $(OBJS) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

%.o: %.c $(DEPS) 
	$(CC) -c -o $@ $< 

classification.bin: classification.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

.PHONY: clean

clean:
	rm -rf */*.o *.o *.bin


