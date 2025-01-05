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

// getopt
#include <stdio.h>  
#include <unistd.h>  

using namespace cv;
using namespace cv::face;

// Computes the average color of a zone delimited by landmarks
Scalar avg_color(Mat &img, const std::vector<Point>& contour)
{
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  fillPoly(mask, contour, Scalar(255, 255, 255));

  // display the mask to check that it is correct
  // imshow("Mask", mask);
  // waitKey(0);
  
  // convert in HSV
  /* Mat img_hsv;
     cvtColor(img, img_hsv, COLOR_BGR2HSV); */

  // average color inside the bounding box
  Scalar avg_bb = mean(img, mask);
  std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  
  return avg_bb;
}

// Computes the average color of a zone delimited by landmarks
Scalar avg_color(Mat &img, const std::vector<std::vector<Point>>& contour)
{
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  fillPoly(mask, contour, Scalar(255, 255, 255));

  // display the mask to check that it is correct
  // imshow("Mask", mask);
  // waitKey(0);
  
  // convert in HSV
  /* Mat img_hsv;
     cvtColor(img, img_hsv, COLOR_BGR2HSV); */

  // average color inside the bounding box
  Scalar avg_bb = mean(img, mask);
  std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  
  return avg_bb;
}

// Corrects the eye color assuming that the average was computed over the circumcircle of the iris
// excludes the eye white around the iris and the black pupil inside
Scalar correct_eye_color(Scalar avg_bb)
{  // corrected for the pupil and the white
  Scalar whitep = (1 - CV_PI/4)*Scalar(255, 255, 255);
  Scalar blackp = CV_PI*Scalar(0, 0, 0)/36;
  Scalar avg = (9/2)*(avg_bb - whitep - blackp)/CV_PI; // correction with the pupil
  Scalar avg2 = 4*(avg_bb - whitep)/CV_PI; // correction without the pupil 
  
  std::cout << "Average color of the bbox = \t" << avg_bb << std::endl;
  std::cout << "Contribution of white = \t" << whitep << std::endl;
  std::cout << "Contribution of black = \t" << blackp << std::endl;
  std::cout << "Actual color of the iris = \t" << avg << std::endl;

  // draw these colors on the image for checking
  /* std::vector<Point> avg_color_contour = {Point(0, 0), Point(0, 40), Point(40, 40), Point(40, 0)};
  fillPoly(img, avg_color_contour, avg_bb);

  std::vector<Point> avg_contour = {Point(40, 0), Point(40, 40), Point(80, 40), Point(80, 0)};
  fillPoly(img, avg_contour, avg);

  std::vector<Point> avg2_contour = {Point(80, 0), Point(80, 40), Point(120, 40), Point(120, 0)};
  fillPoly(img, avg2_contour, avg2); */

  return avg;
}

// prints a color on the image
// for positioning, we know the white band is on the right side, so we just need the height index
void print_color(Mat &img, Scalar c, int height, std::string label)
{
  Size textsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, 0);
  Point pos_txt((img.cols - 120), height*40 + 30);
  std::cout << "The text has a width of " << std::endl;
  putText(img, label, pos_txt, FONT_HERSHEY_SIMPLEX, 80/(float)textsize.width, Scalar(0, 0, 0), 1, LINE_8);
  
  std::vector<Point> contour = {Point(img.cols-40, height*40),
				Point(img.cols-40, (height+1)*40),
				Point(img.cols, (height+1)*40),
				Point(img.cols, height*40)};
  fillPoly(img, {contour}, c);
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

// utility function: convert a color
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

void histogram(
	       Mat &img, const std::vector<std::vector<Point>>& contour,
	       int hbins, int sbins, int vbins, // hue, saturation, value bins
	       int i // number of the histogram we draw (for positioning)
	       )
{
    MatND hist;

    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    // value varies from 0 to 255 if I'm reading the docs correctly
    float vranges[] = {0, 256};
    const float* ranges[] = { hranges, sranges };

    int scale = 10; // how magnified histogram bins are

    // second border if the image is not tall enough to accomodate two histograms
    if (img.rows < sbins*scale*(i+1) or i == 0)
      {
	copyMakeBorder(img, img, 0, 0, 0, hbins*scale, BORDER_CONSTANT,
		       Scalar(255, 255, 255)
		       );
      }

    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    fillPoly(mask, contour, Scalar(255, 255, 255));
    
    int channels[] = {0, 1, 2};
    
    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    
    calcHist(&img_hsv, // image(s)
	     1, // number of images
	     channels, mask, // mask
	     hist, 2, histSize, ranges,
	     true,  // the histogram is uniform, whatever that means
	     false); // clear the histogram at the beginning of the computation
    
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    
    Mat histImg = Mat::zeros(sbins*scale, hbins*scale, CV_8UC3);
    
    for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
	{	
	  float binVal = hist.at<float>(h, s);
	  int intensity = cvRound(binVal*255/maxVal);
	  
	  Scalar hsv_col(cvRound(h*180/hbins),
			 cvRound(s*255/sbins),
			 intensity);
	  Scalar bgr_col = convert_color(hsv_col, COLOR_HSV2BGR);
	  // cvtColor(hsv_col, bgr_col, COLOR_HSV2BGR);
	  
	  rectangle( histImg, Point(h*scale, s*scale),
		     Point( (h+1)*scale - 1, (s+1)*scale - 1),
		     // Scalar::all(intensity),
		     bgr_col,
		     // hsv_col,
		     -1 );
	}

    // Paste the histogram onto the image

    Rect pos(img.cols - histImg.cols, img.rows - histImg.rows,
		 histImg.cols, histImg.rows);
    // if there is already a histogram and we have space to stack the histogram above it
    if (i > 0 and img.rows >= sbins*scale*(i+1))
      pos = Rect(img.cols - histImg.cols, img.rows - (i+1)*histImg.rows,
	       histImg.cols, histImg.rows);
    
    histImg.copyTo(img(pos));
}

int main(int argc, char** argv)
{
  /*
    Processing command line options
    Switch between webcam and command-line provided image
   */

  bool webcam = true; // if false, use an already existing image
  int c;

  while ((c = getopt (argc, argv, "f:")) != -1)
    switch (c)
      {
      case 'f':
        webcam = false;
        break;
      case '?':
        if (optopt == 'f')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
                   "Unknown option character `\\x%x'.\n",
                   optopt);
        return 1;
      default:
        abort ();
      }
  
  Mat img;

  if (webcam)
    {
      /*
	Take picture from camera
      */
      
      VideoCapture cam(0);
    
      while (cam.read(img))
	{
	  imshow("Camera", img);
	  if (waitKey(1) == 27) break;
	} 
    } else {
    /*
      Read image
    */
    
    std::string image_path = samples::findFile(argv[2]);
    img = imread(image_path, IMREAD_COLOR);
  
    if(img.empty())
      {
	std::cout << "Could not read the image: " << image_path << std::endl;
	return 1;
      } 
  }
  
  /*
    Histograms parameters
  */

  // Quantize the hue to 30 levels
  // and the saturation to 32 levels
  int hbins = 30, sbins = 32, vbins = 32;
  
  // we extend the image to be able to display the color info 
  copyMakeBorder(img, img, 0, 0, 0, 120, BORDER_CONSTANT,
		 Scalar(255, 255, 255)
		 );


  /* Retrieve the facial landmarks of the image (do it once because it is costly) */
  std::vector<std::vector<Point2f> > landmarks;
  get_landmarks(img, landmarks);

  /*
    detecting eye color
  */

  /* Left pupil */
  
  std::vector<Point> left_pupil_contour = left_pupil(landmarks[0]);

  // face::drawFacemarks(img, landmarks[0], Scalar(0,255,200));

  Scalar avg_left_pupil = avg_color(img, left_pupil_contour);
  print_color(img, avg_left_pupil, 0, "Left eye");
  
  /* Right pupil */
  
  std::vector<Point> right_pupil_contour = right_pupil(landmarks[0]);
  Scalar avg_right_pupil = avg_color(img, right_pupil_contour);
  print_color(img, avg_right_pupil, 1, "Right eye");

  /*
    Detecting skin tone
  */

  std::vector<Point> face_contour = face_oval(landmarks[0]);

  // removing the features (eyes and mouth)
  std::vector<std::vector<Point>> face_wo_features = {face_contour};

  face_wo_features.push_back({});
  for (int i=47; i >= 42; i--) // right eye
    face_wo_features[1].push_back(landmarks[0][i]); 

  face_wo_features.push_back({});
  for (int i=41; i >= 36; i--) // left eye
    face_wo_features[2].push_back(landmarks[0][i]); 

  face_wo_features.push_back({});
  for (int i=59; i >= 48; i--) // mouth
    face_wo_features[3].push_back(landmarks[0][i]); 

  /* polylines(img,
	    face_wo_features,
	    true,
	    Scalar(255, 200, 0),
	    2,
	    LINE_AA); */

	    
  std::cout << "Computing the average color of the face" << std::endl;
  Scalar avg_face = avg_color(img, face_wo_features);
  print_color(img, avg_face, 2, "Face");

  /*
    Compute the color histogram of the face region

    Mostly lifted from the docs https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
  */

  // we first try with a pure skin mask
  histogram(img, face_wo_features, hbins, sbins, vbins, 0);
  histogram(img, {right_pupil_contour}, hbins, sbins, vbins, 1);  
  
  /* namedWindow( "H-S Histogram", 1 );
  imshow( "H-S Histogram", histImg );

  waitKey(0); */
  
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
  // contours(img, thr);
  // threshold(img_contours, img_contours, i, i*2, THRESH_BINARY | THRESH_OTSU); 

  /* std::string txt = std::to_string(thr);
  cv::Scalar color = Scalar(255, 255, 255) - avg_left_pupil;
  Size textsize = getTextSize(txt, FONT_HERSHEY_SIMPLEX, 1, 1, 0);
  int vPos = (int)(1.3 * textsize.height);
  Point org((img.cols - textsize.width), vPos);
  putText(img, txt, org, FONT_HERSHEY_SIMPLEX, 1, color, 1, LINE_8);*/
  
  // imshow("result", channels[1]);
  imshow("res", img);
  waitKey(0);
  // }
    
  return 0;
}
