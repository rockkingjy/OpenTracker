
CC=gcc

VPATH=./helper:./loader:./network:./tracker:/media/elab/sdd/caffe

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/mycodes/caffe/build/lib -lcaffe  \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn \
	-lboost_system -lboost_filesystem -lboost_regex -lglog 

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x \
	-I/media/elab/sdd/mycodes/caffe/build/include/ -I/media/elab/sdd/mycodes/caffe/include/ \
	-I/usr/local/cuda/include/ 

HEADERS = $(wildcard */*.h)

#$(info DEPS is: $(DEPS))

OBJS = helper/bounding_box.o helper/helper.o helper/high_res_timer.o helper/image_proc.o \
		loader/loader_base.o network/regressor_base.o network/regressor.o \
		tracker/tracker.o rungoturntracker.o

all: rungoturntracker.bin classification.bin

classification.bin: classification.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 
	
rungoturntracker.bin: $(OBJS) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

%.o: %.c $(HEADERS) 
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


.PHONY: clean

clean:
	rm -rf */*.o *.o *.bin
