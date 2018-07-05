
USE_CAFFE=0
USE_CUDA=0

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv`  -lstdc++ -lm \
	-lboost_system -lboost_filesystem -lboost_regex -lglog  -lprotobuf 

CXXFLAGS= -g -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x 
	
HEADERS = $(wildcard *.h)


ifeq ($(USE_CAFFE), 1)
USE_CUDA=1
CXXFLAGS+= -DUSE_CAFFE
LDFLAGS+= -L/media/elab/sdd/mycodes/caffe/build/lib -lcaffe 
CXXFLAGS+= -I/media/elab/sdd/mycodes/caffe/build/include/ -I/media/elab/sdd/mycodes/caffe/include/ 
endif

ifeq ($(USE_CUDA), 1)
CXXFLAGS+= -DUSE_CUDA
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn 
CXXFLAGS+= -I/usr/local/cuda/include/ 
endif

OBJS = fftTool.o fhog.o interpolator.o optimize_scores.o \
		regularization_filter.o feature_extractor.o feature_operator.o  \
		training.o sample_update.o eco.o runecotracker.o

all: runecotracker.bin

runecotracker.bin: $(OBJS) 
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean

clean:
	rm -rf ./.d */*.o *.o *.bin

