all: main.cpp color.o
	g++ main.cpp color.o `pkg-config --cflags --libs opencv4` -lopencv_face -o face

color.o: color.cpp color.hpp
	g++ -c color.cpp `pkg-config --cflags --libs opencv4` -o color.o
