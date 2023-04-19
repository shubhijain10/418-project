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

HEADERS := src/include/*.hpp
SOURCES := src/*.cpp

CXX=g++ -m64
CFLAGS=-std=c++14 -fvisibility=hidden -lpthread

.SUFFIXES:
.PHONY: all clean

all: canny-filter

canny-filter: $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

clean:
	rm -rf ./canny-filter


# canny-filter: $(HEADERS) main.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v1.cpp

# nbody-$(CONFIGURATION)-v2: $(HEADERS) src/mpi-simulator-v2.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v2.cpp

# FILES = src/*.cpp \
# 		src/*.h