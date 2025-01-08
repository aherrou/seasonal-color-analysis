#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Computes the average color of a zone delimited by landmarks
Scalar avg_color(Mat &img, const std::vector<Point>& contour);

// Computes the average color of a zone delimited by landmarks
Scalar avg_color(Mat &img, const std::vector<std::vector<Point>>& contour);

// Corrects the eye color assuming that the average was computed over a square bounding box containing the iris 
// excludes the eye white around the iris and the black pupil inside
Scalar correct_eye_color(Scalar avg_bb);

// prints a color on the image
// for positioning, we know the white band is on the right side, so we just need the height index
void print_color(Mat &img, Scalar c, int height, std::string label);

// utility function: convert a color
Scalar convert_color(Scalar source_col, int mode);
