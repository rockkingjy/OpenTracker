
CC=gcc

VPATH=./helper:./loader:./network:./tracker:/media/elab/sdd/caffe

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/caffe/build/lib -lcaffe-nv  \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn \
	-lboost_system -lboost_filesystem -lboost_regex -lglog  -lprotobuf

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x \
	-I/media/elab/sdd/caffe/.build_release/src/ -I/media/elab/sdd/caffe/include/ \
	-I/usr/local/cuda/include/

DEPS = $(wildcard */*.h) $(wildcard /media/elab/sdd/caffe/include/*/*.hpp)

#$(info DEPS is: $(DEPS))

OBJS = fftTool.o fhog_f.o FHOG.o interpolator.o optimize_scores.o \
		reg_filter.o feature_extractor.o feature_operator.o  \
		training.o eco_sample_update.o eco.o runecotracker.o

all: runecotracker.bin

runecotracker.bin: $(OBJS) 
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

%.o: %.c $(DEPS) 
	$(CC) -c -o $@ $< 

.PHONY: clean

clean:
	rm -rf */*.o *.o *.bin

