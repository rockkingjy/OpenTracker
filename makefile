
OPENPOSE=1


CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/mycodes/caffe/build/lib -lcaffe  \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn \
	-lboost_system -lboost_filesystem -lboost_regex -lglog -lstdc++ -lm 

CXXFLAGS= -g -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x \
	-I/media/elab/sdd/mycodes/caffe/build/include/ -I/media/elab/sdd/mycodes/caffe/include/ \
	-I/usr/local/cuda/include/ # -DUSE_OPENCV

HEADERS = $(wildcard *.h) *.hpp $(wildcard kcf/*.h) $(wildcard goturn/*/*.h) $(wildcard eco/*.h)

OBJ = kcf/fhog.o kcf/kcftracker.o \
	goturn/network/regressor_base.o goturn/network/regressor.o \
	goturn/helper/bounding_box.o goturn/helper/helper.o goturn/helper/image_proc.o \
	goturn/helper/high_res_timer.o goturn/tracker/tracker.o trackerscompare.o \
	eco/fftTool.o eco/fhog.o eco/interpolator.o eco/optimize_scores.o \
	eco/regularization_filter.o eco/feature_extractor.o eco/feature_operator.o  \
	eco/training.o eco/eco_sample_update.o eco/eco.o \
	inputs/readdatasets.o 

ifeq ($(OPENPOSE), 1) 
LDFLAGS+=-lpthread -lopenpose -lgflags
OBJ+=inputs/openpose.o
endif

all: makekcf makegoturn makeeco trackerscompare.bin

trackerscompare.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


makekcf:
	cd kcf && make -j`nproc`

makegoturn:
	cd goturn && make -j`nproc`

makeeco:
	cd eco && make -j`nproc`

.PHONY: clean
.PHONY: cleanroot

cleanroot:
	rm *.o

clean:
	rm -rf */*/*.o */*.o *.o */*.bin *.bin

