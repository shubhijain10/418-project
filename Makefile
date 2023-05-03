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

HEADERS-OPENMP := src-openmp/include/*.hpp
SOURCES-OPENMP := src-openmp/*.cpp

HEADERS-SEQ := src-sequential/include/*.hpp
SOURCES-SEQ := src-sequential/*.cpp

OBJDIR-CUDA=objs-cuda
OBJDIR-OPENMP=objs-openmp
CXX-CUDA=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

CFLAGS := -std=c++14 -fvisibility=hidden -lpthread
CFLAGS += -O2 -fopenmp

OBJS-CUDA=$(OBJDIR-CUDA)/main.o  $(OBJDIR-CUDA)/filters.o
OBJS-OPEMP=$(OBJDIR-OPENMP)/main.o  $(OBJDIR-OPENMP)/filters.o

.SUFFIXES:
.PHONY: all clean

all: canny-cuda canny-openmp canny-seq

dirs-cuda:
		mkdir -p $(OBJDIR-CUDA)/

dirs-openmp: 
		mkdir -p $(OBJDIR-OPENMP)/

canny-cuda: dirs-cuda $(OBJS-CUDA)
	$(CXX-CUDA) $(CXXFLAGS) -o $@ $(OBJS-CUDA) $(LDFLAGS)

canny-openmp: $(SOURCES-OPENMP) $(HEADERS-OPENMP)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES-OPENMP)

canny-seq: $(SOURCES-SEQ) $(HEADERS-SEQ)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES-SEQ)

# canny-openmp: dirs-openmp $(OBJS-OPENMP)
# 	$(CXX) -o $@ $(CFLAGS) $(OBJS_OPENMP)
# canny-seq: $(SOURCES-SEQ) $(HEADERS-SEQ)
# 	$(CXX) -o $@ $(CFLAGS) $(SOURCES-SEQ)

clean:
	rm -rf ./canny-*

$(OBJDIR-CUDA)/%.o: src/%.cpp $(HEADERS-CUDA)
		$(CXX-CUDA) $< $(CXXFLAGS) -c -o $@

$(OBJDIR-CUDA)/%.o: src/%.cu $(HEADERS-CUDA)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

# $(OBJDIR-OPENMP)/%.o: src-openmp/%.cpp $(HEADERS-OPENMP)
# 		$(CXX) $< $(CFLAGS) -c -o $@

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