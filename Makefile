CXX := g++
CXXFLAGS := -std=c++11 -O2 -Wall
TARGET := tslf
SRC := src/tslf.cpp

.PHONY: all run test docs clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	bash scripts/test_summarizer.sh

docs:
	doxygen docs/Doxyfile

clean:
	rm -f $(TARGET) summary_50.txt summary_80.txt summary.txt
	rm -rf docs/doxygen_output
