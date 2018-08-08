#0:no-simd;1:sse;2:neon (comment should not in the same line as the parameter)
USE_SIMD=1

CC=gcc
CXX=g++
LDFLAGS=  -lstdc++ -lm

# with -O3, hog is speed-up with simd
CXXFLAGS= -Wall -std=c++0x -lstdc++ -O3 -msse4
HEADERS = *.h *.hpp
OBJ = convConst.o gradientMex.o chnsTestCpp.o


ifeq ($(USE_SIMD), 1)
CXXFLAGS+= -DUSE_SIMD -msse4
endif

ifeq ($(USE_SIMD), 2)
CXXFLAGS+= -DUSE_SIMD -DUSE_NEON -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto 
endif

all: chnsTestCpp.bin 

chnsTestCpp.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


.PHONY: clean

clean:
	rm -rf *.o *.bin *.so

