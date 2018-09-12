USE_VOT_TRAX=0

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm
CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++ -O3 -fPIC
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
TARGET_LIB = libkcftracker.so
OBJS = fhog.o kcftracker.o

ifeq ($(USE_VOT_TRAX), 1)
CXXFLAGS+= -ltrax
LDFLAGS+= -ltrax 
endif

ALL = runkcftracker.bin $(TARGET_LIB)
ifeq ($(USE_VOT_TRAX), 1)
	ALL+= vot_kcf.bin 
endif

all: $(ALL)

runkcftracker.bin: $(OBJS) runkcftracker.o
	$(CC) -o $@ $^ $(LDFLAGS)

ifeq ($(USE_VOT_TRAX), 1)
vot_kcf.bin: $(OBJS) vot_kcf.o 
	$(CC) -o $@ $^ $(LDFLAGS) 
endif

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