
CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm
CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++ -O3 -fPIC
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
TARGET_LIB = libsorttracker.so
OBJS = kalmantracker.o sort.o

ALL = runsorttracker.bin $(TARGET_LIB)

all: $(ALL)

runsorttracker.bin: $(OBJS) runsorttracker.o
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

.PHONY: install
install: $(TARGET_LIB)
	mkdir -p /usr/local/include/opentracker
	mkdir -p /usr/local/include/opentracker/sort
	cp $(TARGET_LIB) /usr/local/lib
	mkdir -p /usr/local/include/opentracker/sort
	cp *.hpp /usr/local/include/opentracker/sort

.PHONY: uninstall
uninstall: $(TARGET_LIB)
	rm -f -r /usr/local/include/opentracker/sort
	rm -f /usr/local/lib/$(TARGET_LIB)