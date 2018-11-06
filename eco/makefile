USE_CAFFE=0
USE_CUDA=0
#0:no-simd;1:sse-Intel;2:neon-TX2;3:neon-Raspberrypi3
#(comment should not in the same line as the parameter)
USE_SIMD=0
USE_MULTI_THREAD=0
USE_FFTW=0
#USE_OPENMP=1
#USE_EIGEN=1
USE_GTEST=0
USE_VOT_TRAX=0

CAFFE_PATH=/media/elab/sdd/mycodes/caffe
CC=gcc
CXX=g++
NVCC=nvcc
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm 
CXXFLAGS = -g -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x -O3 -fPIC 
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
TARGET_LIB = libecotracker.so
OBJS = ffttools.o fhog.o interpolator.o optimize_scores.o scale_filter.o \
		regularization_filter.o feature_extractor.o feature_operator.o  \
		training.o sample_update.o eco.o metrics.o

ifeq ($(USE_CAFFE), 1)
CXXFLAGS+= -DUSE_CAFFE
LDFLAGS+= -L$(CAFFE_PATH)/build/lib -lcaffe -lglog  #-lprotobuf 
CXXFLAGS+= -I$(CAFFE_PATH)/build/include/ -I$(CAFFE_PATH)/include/ 
LDFLAGS+= -lboost_system -lboost_filesystem -lboost_regex
endif

ifeq ($(USE_CUDA), 1)
CXXFLAGS+= -DUSE_CUDA
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn 
CXXFLAGS+= -I/usr/local/cuda/include/ 
endif

ifeq ($(USE_SIMD), 1)
CXXFLAGS+= -DUSE_SIMD -msse4
OBJS+= gradient.o
endif

ifeq ($(USE_SIMD), 2)
CXXFLAGS+= -DUSE_SIMD -DUSE_NEON -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto 
OBJS+= gradient.o
endif

ifeq ($(USE_SIMD), 3)
CXXFLAGS+= -DUSE_SIMD -DUSE_NEON -ffast-math -flto -mfpu=neon
OBJS+= gradient.o
endif

ifeq ($(USE_MULTI_THREAD), 1)
CXXFLAGS+= -DUSE_MULTI_THREAD
LDFLAGS+= -pthread
endif

ifeq ($(USE_FFTW), 1)
CXXFLAGS+= -DUSE_FFTW
LDFLAGS+= -lfftw3
endif

#ifeq ($(USE_OPENMP), 1)
#CXXFLAGS+= -DUSE_OPENMP -fopenmp
#LDFLAGS+= -fopenmp
#endif

#ifeq ($(USE_EIGEN), 1)
#CXXFLAGS+= -DUSE_EIGEN -I/usr/include/eigen3
#endif

ifeq ($(USE_GTEST), 1)
GTEST_DIR = ../googletest
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
CXXFLAGS += -isystem $(GTEST_DIR)/include
endif

ifeq ($(USE_VOT_TRAX), 1)
CXXFLAGS+= -ltrax
LDFLAGS+= -ltrax 
endif

ALL+= runecotracker.bin $(TARGET_LIB)
ifeq ($(USE_GTEST), 1) 
	ALL+= eco_unittest.bin 
endif
ifeq ($(USE_VOT_TRAX), 1)
	ALL+= vot_eco.bin 
endif

all: $(ALL)

runecotracker.bin: $(OBJS) runecotracker.o
	$(CC) -o $@ $^ $(LDFLAGS) 

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -shared -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cc $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(ARCH) --compiler-options "$(CXXFLAGS)" -c $< -o $@


ifeq ($(USE_VOT_TRAX), 1)
vot_eco.bin: $(OBJS) vot_eco.o 
	$(CC) -o $@ $^ $(LDFLAGS) 
endif

#gtest=========
ifeq ($(USE_GTEST), 1)
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^
#-------------
eco_unittest.o : eco_unittest.cc ffttools.hpp $(GTEST_HEADERS)
	$(CXX) $(CXXFLAGS) -c eco_unittest.cc

eco_unittest.bin : ffttools.o eco_unittest.o gtest_main.a
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lpthread
endif

#cv gpu test========
cvgputest.o : cvgputest.cc 
	$(CXX) $(CXXFLAGS) -c cvgputest.cc

cvgputest.bin : cvgputest.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

#==============
.PHONY: clean 
clean:
	rm -rf ./.d *.o *.bin *.so *.a */*.o */*.bin

.PHONY: install
install: $(TARGET_LIB)
	mkdir -p /usr/local/include/opentracker
	mkdir -p /usr/local/include/opentracker/eco
	mkdir -p /usr/local/include/opentracker/eco/look_tables
	cp $(TARGET_LIB) /usr/local/lib
	cp *.hpp /usr/local/include/opentracker/eco
	cp *.h /usr/local/include/opentracker/eco
	cp -a look_tables/. /usr/local/include/opentracker/eco/look_tables

.PHONY: uninstall
uninstall: $(TARGET_LIB)
	rm -f -r /usr/local/include/opentracker/eco
	rm -f /usr/local/lib/$(TARGET_LIB)