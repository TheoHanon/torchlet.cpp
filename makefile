.PHONY : clean
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Iinclude

main : test.o tensor.o init.o linear.o kernel.o
	$(CXX) $(CXXFLAGS) test.o tensor.o init.o linear.o kernel.o -o test

test.o : test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp -o test.o

tensor.o: src/tensor.cpp
	$(CXX) $(CXXFLAGS) -c src/tensor.cpp -o tensor.o

init.o: src/init.cpp
	$(CXX) $(CXXFLAGS) -c src/init.cpp -o init.o

linear.o : src/linear.cpp
	$(CXX) $(CXXFLAGS) -c src/linear.cpp -o linear.o

kernel.o : src/kernel.cpp
	$(CXX) $(CXXFLAGS) -c src/kernel.cpp -o kernel.o

clean:
	rm -f *.o test