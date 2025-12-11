CXX = g++
CXXFLAGS = -O3 -mavx -std=c++17

SRCS = Matrix.cpp NeuralNetwork.cpp utils.cpp main.cpp
TARGET = nn

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
