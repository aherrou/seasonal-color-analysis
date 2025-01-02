#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/face/facemarkLBF.hpp>
// #include <opencv2/face/facemark.hpp>
// #include <opencv2/face/facemark_train.hpp>
// #include <opencv2/face/face_alignment.hpp>

#include <iostream>

using namespace cv;
using namespace cv::face;

// Computes the average color of a zone delimited by landmarks
Scalar avg_color(Mat &img, const std::vector<Point>& contour)
{
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  fillPoly(mask, contour, Scalar(255, 255, 255));
  
  // convert in HSV
  /* Mat img_hsv;
     cvtColor(img, img_hsv, COLOR_BGR2HSV); */

  // average color inside the bounding box
  Scalar avg_bb = mean(img, mask);

  // corrected for the pupil and the white
  Scalar whitep = (1 - CV_PI/4)*Scalar(255, 255, 255);
  Scalar blackp = CV_PI*Scalar(0, 0, 0)/36;
  Scalar avg = (9/2)*(avg_bb - whitep - blackp)/CV_PI; // correction with the pupil
  Scalar avg2 = 4*(avg_bb - whitep)/CV_PI; // correction without the pupil 
  
  std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  std::cout << "Contribution of white = \t" << whitep << std::endl;
  std::cout << "Contribution of black = \t" << blackp << std::endl;
  std::cout << "Actual color of the iris = \t" << avg << std::endl;

  std::vector<Point> avg_color_contour = {Point(0, 0), Point(0, 40), Point(40, 40), Point(40, 0)};
  fillPoly(img, avg_color_contour, avg_bb);

  std::vector<Point> avg_contour = {Point(40, 0), Point(40, 40), Point(80, 40), Point(80, 0)};
  fillPoly(img, avg_contour, avg);

  std::vector<Point> avg2_contour = {Point(80, 0), Point(80, 40), Point(120, 40), Point(120, 0)};
  fillPoly(img, avg2_contour, avg2);

  return avg;
}

// compute the facial landmarks on img and store them in landmarks
bool get_landmarks(const Mat &img, std::vector<std::vector<Point2f>> &landmarks)
{
  /*create the facemark instance*/
  FacemarkLBF::Params params;
  params.model_filename = "lbfmodel.yaml"; // the trained model will be saved using this filename
  Ptr<Facemark> facemark = FacemarkLBF::create(params);

  facemark->loadModel(params.model_filename);
  
  // Load Face Detector
  CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
  
  std::cout << "Detecting faces and landmarks" << std::endl;
    
  std::vector<cv::Rect> faces;
  Mat gray;
  // Convert frame to grayscale because
  // faceDetector requires grayscale image.
  cvtColor(img, gray, COLOR_BGR2GRAY);
  
  // Detect faces
  faceDetector.detectMultiScale(gray, faces);
  
  bool success = facemark->fit(img, faces, landmarks);

  return success;
}

// draw the contours of a zone of the image (eyes for example)
void contours(const Mat &img, int thr)
{
  // thresholding necessitates a monochrome image
   Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY); 

  /* Mat img_hsv;
  cvtColor(img, img_hsv, COLOR_BGR2HSV);
  // use only one channel 
  Mat channels[3];
  split(img_hsv, channels); */

  std::cout << "Finding contours for threshold " << thr << std::endl;

  Mat thresh;
  threshold(gray, thresh, thr, 255, THRESH_BINARY);
  
  std::vector<std::vector<Point> > contours;
  std::vector<Vec4i> hierarchy;
  findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
  drawContours(img, contours, -1, Scalar(0, 200, 200));

  return;
}

// extracts the border of the left pupil from the landmarks of a face
// TODO: maybe rewrite with a template for Point/Point2f
std::vector<Point> left_pupil(const std::vector<Point2f> &landmarks){
  return {landmarks[37], landmarks[38], landmarks[40], landmarks[41]};
}

std::vector<Point> right_pupil(const std::vector<Point2f> &landmarks){
  return {landmarks[43], landmarks[44], landmarks[46], landmarks[47]};
}

int main(int argc, char** argv)
{
  // read image 
  std::string image_path = samples::findFile(argv[1]);
  Mat img = imread(image_path, IMREAD_COLOR);
  
  if(img.empty())
    {
      std::cout << "Could not read the image: " << image_path << std::endl;
      return 1;
    }

  /* Retrieve the facial landmarks of the image (do it once because it is costly) */
  std::vector<std::vector<Point2f> > landmarks;
  get_landmarks(img, landmarks);

  /*
    detecting eye color
  */
  
  // according to https://forum.opencv.org/t/is-there-any-good-face-shape-detection-solution-based-on-opencv/5663/2
  // left eye is landmarks [36; 41] and right eye is landmarks [42;47]
  // the pupils are {37; 38; 40; 41} and {43; 44; 46; 47}


  /* Left pupil */
  
  std::vector<Point> left_pupil_contour = left_pupil(landmarks[0]);

  // rectangle(left_pupil_mask, left_pupil_bbox.tl(), left_pupil_bbox.br(), Scalar(255), FILLED); 
  Mat left_pupil_mask = Mat::zeros(img.size(), CV_8UC1); // 8UC3 necessary because we want to "and "

  // create a polygon containing the left pupil on left_pupil_mask
  // fillPoly(left_pupil_mask, left_pupil_contour, Scalar(255, 255, 255));

  // masking the face image by this polygon using bitwise and
  // Mat temp = img & left_pupil_mask; 

  // face::drawFacemarks(img, landmarks[0], Scalar(0,255,200));

  // now we compute the average color inside the mask

  Scalar avg_left_pupil = avg_color(img, left_pupil_contour);
  
  // and we draw the mask in this color on the 
  // fillPoly(img, left_pupil_contour, avg_left_pupil);

  copyMakeBorder(img, img, 0, 0, 0, 100, BORDER_CONSTANT,
		 avg_left_pupil // Scalar(255, 255, 255)
		 );

  // write that this is the left pupil
  
  
  /* Right pupil */
  
  /* std::vector<Point> right_pupil_contour = right_pupil(landmarks[0]);
  
  // polylines(img, right_pupil_contour, true, Scalar(255, 200, 0), 2, FILLED);

  // std::cout << right_pupil_contour << std::endl;
  // face::drawFacemarks(img, right_pupil_contour, Scalar(255,0,200));
 
  Mat right_pupil_mask = Mat::zeros(img.size(), CV_8UC3);
  fillPoly(right_pupil_mask, right_pupil_contour, Scalar(255, 255, 255));

  // masking the face image by this polygon using bitwise and
  Mat temp2 = img & right_pupil_mask; */

  /*
    Display of the result
  */

  // Mat res = temp | temp2;

  /* Mat img_hsv;
  cvtColor(img, img_hsv, COLOR_BGR2HSV);
  Mat channels[3];
  split(img_hsv, channels);*/

  // for (int i=0; i<255; i+=10){
  //  int i = 2;
  // Mat img_contours = img.clone();
  // Mat img_contours; cvtColor(img, img_contours, COLOR_BGR2GRAY);

  int thr = 100;
  contours(img, thr);
  // threshold(img_contours, img_contours, i, i*2, THRESH_BINARY | THRESH_OTSU); 

  std::string txt = std::to_string(thr);
  cv::Scalar color = Scalar(255, 255, 255) - avg_left_pupil;
  Size textsize = getTextSize(txt, FONT_HERSHEY_SIMPLEX, 1, 1, 0);
  int vPos = (int)(1.3 * textsize.height);
  Point org((img.cols - textsize.width), vPos);
  putText(img, txt, org, FONT_HERSHEY_SIMPLEX, 1, color, 1, LINE_8);
  
  // imshow("result", channels[1]);
  imshow(txt, img);
  waitKey(0);
  // }
  
  return 0;
}
