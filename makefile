
CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/mycodes/caffe/build/lib -lcaffe  \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn \
	-lboost_system -lboost_filesystem -lboost_regex -lglog 

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x \
	-I/media/elab/sdd/mycodes/caffe/build/include/ -I/media/elab/sdd/mycodes/caffe/include/ \
	-I/usr/local/cuda/include/ # -DUSE_OPENCV

DEPS = *.h *.hpp $(wildcard kcf/*.h) $(wildcard goturn/*/*.h) $(wildcard eco/*.h)

OBJ = kcf/fhog.o kcf/kcftracker.o \
	goturn/network/regressor_base.o goturn/network/regressor.o \
	goturn/helper/bounding_box.o goturn/helper/helper.o goturn/helper/image_proc.o \
	goturn/helper/high_res_timer.o goturn/tracker/tracker.o trackerscompare.o \
	eco/fftTool.o eco/fhog.o eco/interpolator.o eco/optimize_scores.o \
	eco/reg_filter.o eco/feature_extractor.o eco/feature_operator.o  \
	eco/training.o eco/eco_sample_update.o eco/eco.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< 

all: makekcf makegoturn makeeco trackerscompare.bin

trackerscompare.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

makekcf:
	cd kcf && make -j4

makegoturn:
	cd goturn && make -j8

makeeco:
	cd eco && make -j8

.PHONY: clean

clean:
	rm -rf */*/*.o */*.o *.o */*.bin *.bin

