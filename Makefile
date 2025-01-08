all: main.cpp
	g++ main.cpp `pkg-config --cflags --libs opencv4` -lopencv_face -o face

