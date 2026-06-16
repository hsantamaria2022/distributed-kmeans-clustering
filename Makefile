CXX = mpicxx
CXXFLAGS = -fopenmp -O2 -std=c++11

all: kmeans generate_data

kmeans: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

generate_data: generateRndSeeds.cpp
	g++ -O2 -std=c++11 -o $@ $<

clean:
	rm -f kmeans generate_data

.PHONY: all clean
