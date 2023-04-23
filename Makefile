OUTPUTDIR := bin/

# CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -Wall -Wextra

# ifeq (,$(CONFIGURATION))
# 	CONFIGURATION := release
# endif

# ifeq (debug,$(CONFIGURATION))
# CFLAGS += -g
# else
# CFLAGS += -O2
# endif

# HEADERS-SEQ := src-sequential/include/*.hpp
# SOURCES-SEQ := src-sequential/*.cpp

HEADERS-CUDA := src/include/*.hpp
SOURCES-CUDA := src/*.cpp

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/filters.o

.SUFFIXES:
.PHONY: all clean

all: canny-cuda

dirs:
		mkdir -p $(OBJDIR)/

canny-cuda: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# canny-seq: $(SOURCES-SEQ) $(HEADERS-SEQ)
# 	$(CXX) -o $@ $(CFLAGS) $(SOURCES-SEQ)

clean:
	rm -rf ./canny-*

$(OBJDIR)/%.o: src/%.cpp $(HEADERS-CUDA)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: src/%.cu $(HEADERS-CUDA)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@


# canny-filter: $(HEADERS) main.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v1.cpp

# nbody-$(CONFIGURATION)-v2: $(HEADERS) src/mpi-simulator-v2.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v2.cpp

# FILES = src/*.cpp \
# 		src/*.h

# .SUFFIXES:
# .PHONY: all clean

# all: canny-seq canny-cuda

# dirs:
# 		mkdir -p $(OBJDIR)/

# canny-cuda: 

# canny-seq: $(SOURCES-SEQ) $(HEADERS-SEQ)
# 	$(CXX) -o $@ $(CFLAGS) $(SOURCES-SEQ)