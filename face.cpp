#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>

using namespace cv;
using namespace cv::face;

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

void contours(const Mat &img, int thr)
{
  // thresholding necessitates a monochrome image
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY); 

  std::cout << "Finding contours for threshold " << thr << std::endl;

  Mat thresh;
  threshold(gray, thresh, thr, 255, THRESH_BINARY);
  
  std::vector<std::vector<Point> > contours;
  std::vector<Vec4i> hierarchy;
  findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
  drawContours(img, contours, -1, Scalar(0, 200, 200));

  return;
}

// according to https://forum.opencv.org/t/is-there-any-good-face-shape-detection-solution-based-on-opencv/5663/2
// left eye is landmarks [36; 41] and right eye is landmarks [42;47]
// the pupils are {37; 38; 40; 41} and {43; 44; 46; 47}

// extracts the border of the left pupil from the landmarks of a face
// TODO: maybe rewrite with a template for Point/Point2f
std::vector<Point> left_pupil(const std::vector<Point2f> &landmarks){
  return {landmarks[37], landmarks[38], landmarks[40], landmarks[41]};
}

std::vector<Point> right_pupil(const std::vector<Point2f> &landmarks){
  return {landmarks[43], landmarks[44], landmarks[46], landmarks[47]};
}

std::vector<Point> face_oval(const std::vector<Point2f> &landmarks){
  std::vector<Point> res = {};

  // cheeks and jaw
  for (int i=0; i<=16; i++)
    res.push_back(landmarks[i]);

  // forehead
  for (int i=26; i>=17; i--)
    res.push_back(landmarks[i]);
    
  return res;
}
