#include "color.hpp"

Scalar avg_color(Mat &img, const std::vector<Point>& contour)
{
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  fillPoly(mask, contour, Scalar(255, 255, 255));

  // polylines(img, contour, true, Scalar(255, 255, 0), 1);
  
  // average color inside the bounding box
  Scalar avg_bb = mean(img, mask);
  // std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  
  return avg_bb;
}

Scalar avg_color(Mat &img, const std::vector<std::vector<Point>>& contour)
{
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  fillPoly(mask, contour, Scalar(255, 255, 255));

  // polylines(img, contour, true, Scalar(255, 255, 0), 1);

  // average color inside the bounding box
  Scalar avg_bb = mean(img, mask);
  // std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  
  return avg_bb;
}

Scalar correct_eye_color(Scalar avg_bb)
{  // corrected for the pupil and the white
  Scalar whitep = (1 - CV_PI/4)*Scalar(255, 255, 255);
  Scalar blackp = CV_PI*Scalar(0, 0, 0)/36;
  Scalar avg = (9/2)*(avg_bb - whitep - blackp)/CV_PI; // correction with the pupil
  Scalar avg2 = 4*(avg_bb - whitep)/CV_PI; // correction without the pupil 
  
  /* std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  std::cout << "Contribution of white = \t" << whitep << std::endl;
  std::cout << "Contribution of black = \t" << blackp << std::endl;
  std::cout << "Actual color of the iris = \t" << avg << std::endl; */

  // draw these colors on the image for checking
  /* std::vector<Point> avg_color_contour = {Point(0, 0), Point(0, 40), Point(40, 40), Point(40, 0)};
  fillPoly(img, avg_color_contour, avg_bb);

  std::vector<Point> avg_contour = {Point(40, 0), Point(40, 40), Point(80, 40), Point(80, 0)};
  fillPoly(img, avg_contour, avg);

  std::vector<Point> avg2_contour = {Point(80, 0), Point(80, 40), Point(120, 40), Point(120, 0)};
  fillPoly(img, avg2_contour, avg2); */

  return avg;
}

void print_color(Mat &img, Scalar c, int height, std::string label)
{
  Size textsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, 0);
  Point pos_txt((img.cols - 120), height*40 + 30);

  putText(img, label, pos_txt, FONT_HERSHEY_SIMPLEX, 80/(float)textsize.width, Scalar(0, 0, 0), 1, LINE_8);
  
  std::vector<Point> contour = {Point(img.cols-40, height*40),
				Point(img.cols-40, (height+1)*40),
				Point(img.cols, (height+1)*40),
				Point(img.cols, height*40)};
  fillPoly(img, {contour}, c);
}

Scalar convert_color(Scalar source_col, int mode)
{
  Mat source(1, 1, CV_8UC3, source_col);
  Mat target;
  cvtColor(source, target, mode);
  // return Scalar(target[0], target[1], target[2]);
  return Scalar((int)target.at<cv::Vec3b>(0, 0)[0],
	 (int)target.at<cv::Vec3b>(0, 0)[1],
	 (int)target.at<cv::Vec3b>(0, 0)[2]);
}
