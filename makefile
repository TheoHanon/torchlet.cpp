.PHONY : clean
CXX = clang++
CXXFLAGS = -std=c++17 -Wall

main : test.o tensor.o
	$(CXX) $(CXXFLAGS) test.o tensor.o -o test

test.o : test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp -o test.o

tensor.o: src/tensor.cpp
	$(CXX) $(CXXFLAGS) -c src/tensor.cpp -o tensor.o

clean:
	rm -f *.o test