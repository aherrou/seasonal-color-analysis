# Seasonal color analysis

This projects aims at implementing the aesthetic theory of [seasonal color analysis](https://gabriellearruda.com/seasonal-color-analysis-what-season-are-you/) 
in order to provide automatic personal style recommendation from a single picture. 

## Building

This project is implemented using [OpenCV](https://opencv.org/). Build it with `opencv_contrib` following the instructions at [https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html].

The project can be built by simply runnig `make`. 

## Usage

**Warning:** this program assumes that the picture provided has been taken in neutral (natural) lighting, 
with uniform exposure all over the face and minimal reflexion on the iris of the eye.

Two options are available: using a locally saved picture or taking a picture from the camera.

To compute the color analysis of a picture taken from the webcam, just run `./face`.
A webcam window will open, press Esc when the picture is satisfying. 

To use a file, run `./face -f <file>`.

## Roadmap

- [x] Detect faces and segment them in skin, eyes, mouth
- [x] Compute skin and eye color histograms 
- [ ] Determine skin warmth from the histogram
- [ ] Determine saturation of coloring (muted/bright)
- [ ] Determine face color contrast 
- [ ] Compute flattering color palette from warmth, contrast and saturation values
- [ ] Add alternative methods for computing color palettes
