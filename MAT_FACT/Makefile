PROG = bench.exe

CXX = g++

CXXFLAGS += $(USER_DEFINES) -std=c++0x -fopenmp
LDFLAGS += $(USER_DEFINES) -std=c++0x -fopenmp

OBJS = template.o kernel.o

all: $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $(PROG) -std=c++0x	

clean:
	rm -f $(PROG) $(OBJS)
