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

HEADERS := src/*.h

CXX=g++ -m64
CXXFLAGS=-Wall -g

.SUFFIXES:
.PHONY: all clean

all: canny-filter

canny-filter: main.o
	$(CXX) $(CXXFLAGS) -o $@ canny-filter main.o

main.o: src/main.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@



# canny-filter: $(HEADERS) main.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v1.cpp

# nbody-$(CONFIGURATION)-v2: $(HEADERS) src/mpi-simulator-v2.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v2.cpp

clean:
	rm -rf ./canny-filter

FILES = src/*.cpp \
		src/*.h