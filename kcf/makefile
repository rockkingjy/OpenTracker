
CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm

CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++

HEADERS = *.h *.hpp

OBJ = fhog.o kcftracker.o runkcftracker.o

runkcftracker.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


.PHONY: clean

clean:
	rm -rf *.o *.bin

