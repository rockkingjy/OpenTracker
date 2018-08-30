
CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm

CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++ -fPIC

HEADERS = *.h *.hpp

TARGET_LIB = libkcftracker.so

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


.PHONY: install
install: $(TARGET_LIB)
	mkdir -p /usr/local/include/opentracker
	mkdir -p /usr/local/include/opentracker/kcf
	cp $(TARGET_LIB) /usr/local/lib
	mkdir -p /usr/local/include/opentracker/kcf
	cp *.hpp /usr/local/include/opentracker/kcf


.PHONY: uninstall
uninstall: $(TARGET_LIB)
	rm -f -r /usr/local/include/opentracker/kcf
	rm -f /usr/local/lib/$(TARGET_LIB)