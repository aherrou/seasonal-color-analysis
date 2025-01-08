#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// getopt
#include <stdio.h>  
#include <unistd.h>  

#include "color.hpp"
#include "face.hpp"

using namespace cv;
// using namespace cv::face;

// face parts

enum class FacePart {
  FACE,
  EYES,
  MOUTH
};

void histogram(
	       Mat &img, const std::vector<std::vector<Point>>& contour,
	       int hbins, int sbins, int vbins, // hue, saturation, value bins
	       int i, // number of the histogram we draw (for positioning)
	       std::string label,
	       FacePart fp = FacePart::FACE,
	       bool save = false,
	       std::string path = "."
	       )
{
  /*
    Compute the color histogram of the face region
    
    Mostly lifted from the docs https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
  */
  
  MatND hist;
  
  int histSize[] = {hbins, sbins};
  // hue varies from 0 to 179, see cvtColor

  // differentiated between face parts 
  float hmax;
  if (fp == FacePart::FACE){
    hmax = 30;   // [0; 60] are the yellow-orange-red hues aka skin tones
    std::cout << "Displaying the face histogram" << std::endl;
  }
  else
    hmax = 180;
  
  float hranges[] = { 0, hmax };
  // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
  float sranges[] = { 0, 256 };
  // value varies from 0 to 255 if I'm reading the docs correctly
  float vranges[] = {0, 256};
  const float* ranges[] = { hranges, sranges };
  
  int scale = 10; // how magnified histogram bins are
  
  // second border if the image is not tall enough to accomodate two histograms
  // warning: this assumes that all histograms have the same number of bins
  if (img.rows < (sbins+1)*scale*(i+1) or i == 0)
    {
      copyMakeBorder(img, img, 0, 0, 0, (hbins+1)*scale, BORDER_CONSTANT,
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

  Size textsize  = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, 0);
  
  Mat histImg = Mat::zeros((sbins + 1)*scale + textsize.height, (hbins+1)*scale, CV_8UC3);
  
  for( int h = 0; h < hbins; h++ )
    for( int s = 0; s < sbins; s++ )
      {	
	float binVal = hist.at<float>(h, s);
	int intensity = cvRound(binVal*255/maxVal);
	
	Scalar hsv_col(cvRound(h*hranges[1]/hbins),
		       cvRound(s*sranges[1]/sbins),
		       // 255,
		       intensity);
	Scalar bgr_col = convert_color(hsv_col, COLOR_HSV2BGR);
	// cvtColor(hsv_col, bgr_col, COLOR_HSV2BGR);
	
	rectangle( histImg, Point(h*scale, s*scale),
		   Point( (h+1)*scale - 1, (s+1)*scale - 1),
		   bgr_col,
		   -1 );
      }

  // Display the legend

  // hue
  for (int h=0; h<hbins; h++)
    {
      Scalar hsv_col(
		     cvRound(h*hranges[1]/hbins),
		     255,
		     255
		     );
	Scalar bgr_col = convert_color(hsv_col, COLOR_HSV2BGR);
	// cvtColor(hsv_col, bgr_col, COLOR_HSV2BGR);
	
	rectangle( histImg, Point(h*scale, sbins*scale),
		   Point( (h+1)*scale - 1, (sbins+1)*scale - 1),
		   bgr_col,
		   -1 );
    }

  // saturation
  for (int s=0; s<sbins; s++)
    {
      Scalar hsv_col(
		     60,
		     cvRound(s*sranges[1]/sbins),
		     255
		     );
	Scalar bgr_col = convert_color(hsv_col, COLOR_HSV2BGR);
	// cvtColor(hsv_col, bgr_col, COLOR_HSV2BGR);
	
	rectangle( histImg, Point(hbins*scale, s*scale),
		   Point( (hbins+1)*scale - 1, (s+1)*scale - 1),
		   bgr_col,
		   -1 );
    }

  // label
  Point pos_txt(0, histImg.rows - 5);
  putText(histImg, label, pos_txt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1, LINE_8);

  // Paste the histogram onto the image
  
  Rect pos(img.cols - histImg.cols, img.rows - histImg.rows,
	   histImg.cols, histImg.rows);
  // if there is already a histogram and we have space to stack the histogram above it
  if (i > 0 and img.rows >= histImg.rows*(i+1))
    pos = Rect(img.cols - histImg.cols, img.rows - (i+1)*histImg.rows,
	       histImg.cols, histImg.rows);
  
  histImg.copyTo(img(pos));

  // save the histogram?
  if (save)
    {
      bool success = imwrite(path+label+".png", histImg);
      if (success) std::cout << "Wrote image" << std::endl;
      else std::cout << "Fail" << std::endl;
    }
}

int main(int argc, char** argv)
{
  /*
    Processing command line options
    Switch between webcam and command-line provided image
   */

  bool webcam = true; // if false, use an already existing image
  bool save = true; // save the histogram?
  int c;

  while ((c = getopt (argc, argv, "f:")) != -1)
    switch (c)
      {
      case 'f':
        webcam = false;
        break;
	/* case 's':
	save = true;
	break; */
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

  //  if (save) std::cout << "Save the histograms" << std::endl;
  
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
	    
  std::cout << "Computing the average color of the face" << std::endl;
  Scalar avg_face = avg_color(img, face_wo_features);
  print_color(img, avg_face, 2, "Face");

  // we first try with a pure skin mask
  histogram(img, face_wo_features, hbins, sbins, vbins, 0, "Face", FacePart::FACE, true, "./hists/");
  // histogram(img, {right_pupil_contour}, hbins, sbins, vbins, 1, "Eyes", FacePart::EYES);  
  
  /*
    Display of the result
  */

  imshow("res", img);
  waitKey(0);
    
  return 0;
}
