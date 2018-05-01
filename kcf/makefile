

CC=gcc

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm

CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++

DEPS = *.h *.hpp

OBJ = fhog.o kcftracker.o runtracker.o


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< 

runtracker.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)


.PHONY: clean

clean:
	rm -rf *.o *.bin
