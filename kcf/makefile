
CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm

CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++

HEADERS = *.h *.hpp

TARGET_LIB = libkcf.so # target lib

OBJ = fhog.o kcftracker.o runkcftracker.o

all: runkcftracker.bin $(TARGET_LIB)

runkcftracker.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -shared -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)


.PHONY: clean

clean:
	rm -rf *.o *.bin *.so

