USE_CAFFE=0
USE_CUDA=0
USE_BOOST=0
USE_SIMD=1
USE_MULTI_THREAD=1
OPENPOSE=0

CAFFE_PATH=/media/elab/sdd/mycodes/caffe

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm

CXXFLAGS= -g -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x -O3

HEADERS = $(wildcard *.h) *.hpp $(wildcard kcf/*.h) $(wildcard eco/*.h)

OBJS=kcf/fhog.o \
	kcf/kcftracker.o \
	eco/ffttools.o \
	eco/fhog.o \
	eco/interpolator.o \
	eco/optimize_scores.o \
	eco/regularization_filter.o \
	eco/feature_extractor.o \
	eco/feature_operator.o  \
	eco/training.o \
	eco/sample_update.o \
	eco/eco.o \
	inputs/readdatasets.o inputs/readvideo.o \
	trackerscompare.o 


ifeq ($(USE_CAFFE), 1)
CXXFLAGS+= -DUSE_CAFFE
LDFLAGS+= -L$(CAFFE_PATH)/build/lib -lcaffe -lglog 
CXXFLAGS+= -I$(CAFFE_PATH)/build/include/ -I$(CAFFE_PATH)/include/ 
HEADERS+= $(wildcard goturn/*/*.h)
OBJS+=goturn/network/regressor_base.o goturn/network/regressor.o \
	goturn/helper/bounding_box.o goturn/helper/helper.o goturn/helper/image_proc.o \
	goturn/helper/high_res_timer.o goturn/tracker/tracker.o
endif

ifeq ($(USE_CUDA), 1)
CXXFLAGS+= -DUSE_CUDA
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn 
CXXFLAGS+= -I/usr/local/cuda/include/ 
endif

ifeq ($(USE_BOOST), 1)
LDFLAGS+= -lboost_system -lboost_filesystem -lboost_regex
endif

ifeq ($(OPENPOSE), 1) 
LDFLAGS+=-lpthread -lopenpose -lgflags
OBJS+=inputs/openpose.o
endif

ifeq ($(USE_SIMD), 1)
CXXFLAGS+= -DUSE_SIMD -msse4
HEADERS+= $(wildcard eco/hog/*.hpp)
OBJS+= eco/gradient.o
endif

ifeq ($(USE_SIMD), 2)
CXXFLAGS+= -DUSE_SIMD -DUSE_NEON -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto 
HEADERS+= $(wildcard eco/hog/*.hpp)
OBJS+= eco/gradient.o
endif

ifeq ($(USE_SIMD), 3)
CXXFLAGS+= -DUSE_SIMD -DUSE_NEON -ffast-math -flto -mfpu=neon
HEADERS+= $(wildcard eco/hog/*.hpp)
OBJS+= eco/gradient.o
endif

ifeq ($(USE_MULTI_THREAD), 1)
CXXFLAGS+= -DUSE_MULTI_THREAD
LDFLAGS+= -pthread
endif

ALL+= makekcf makeeco trackerscompare.bin
ifeq ($(USE_CAFFE), 1) 
	ALL+= makegoturn 
endif

all: $(ALL)

trackerscompare.bin: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


makekcf:
	cd kcf && make -j`nproc`

ifeq ($(USE_CAFFE), 1) 
makegoturn:
	cd goturn && make -j`nproc`
endif

makeeco:
	cd eco && make -j`nproc`

.PHONY: clean
.PHONY: cleanroot

cleanroot:
	rm *.o inputs/*.o *.bin

clean:
	rm -rf */*/*.o */*.o *.o */*.bin *.bin *.so */*.so

