
CC=gcc
CXX=g++

LDFLAGS=  -lstdc++ -lm

CXXFLAGS= -Wall -std=c++0x -lstdc++ -msse4

HEADERS = *.h *.hpp

OBJ = convConst.o gradientMex.o chnsTestCpp.o

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

