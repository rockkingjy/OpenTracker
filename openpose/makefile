
CC=gcc
CXX=g++

#VPATH=./helper:./loader:./network:./tracker:/media/elab/sdd/caffe

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/mycodes/caffe/build/lib -lcaffe   \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn \
	-lboost_system -lboost_filesystem -lboost_regex -lglog  -lprotobuf \
	-lstdc++ -lm -lpthread -lopenpose -lgflags

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x \
	-I/media/elab/sdd/mycodes/caffe/build/include/ -I/media/elab/sdd/mycodes/caffe/include/ \
	-I/usr/local/cuda/include/ 

HEADERS = $(wildcard *.h)

#$(info DEPS is: $(DEPS))

OBJS = openpose_example.o

all: openpose_example.bin

openpose_example.bin: $(OBJS) 
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean

clean:
	rm -rf ./.d */*.o *.o *.bin

