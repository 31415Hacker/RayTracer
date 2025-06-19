# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
LDFLAGS := -ldl -lglfw -lGLEW -lGL

# Executable name
TARGET := main

# Source files
SRCS := src/main.cpp
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean