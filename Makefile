all: main.cpp color.o face.o
	g++ main.cpp color.o face.o `pkg-config --cflags --libs opencv4` -lopencv_face -o face

color.o: color.cpp color.hpp
	g++ -c color.cpp `pkg-config --cflags --libs opencv4` -o color.o

face.o: face.cpp face.hpp
	g++ -c face.cpp `pkg-config --cflags --libs opencv4` -o face.o
