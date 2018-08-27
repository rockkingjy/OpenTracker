
USE_MULTI_THREAD=1

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lopentracker
CXXFLAGS = -Wall -std=c++0x

ifeq ($(USE_MULTI_THREAD), 1)
CXXFLAGS+= -DUSE_MULTI_THREAD
endif

all: run_eco_example.bin

run_eco_example.bin: run_eco_example.o
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cc
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	rm -rf ./.d *.o *.bin *.so *.a */*.o */*.bin
