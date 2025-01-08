// compute the facial landmarks on img and store them in landmarks
bool get_landmarks(const Mat &img, std::vector<std::vector<Point2f>> &landmarks);

// draw the contours of a zone of the image (eyes for example)
void contours(const Mat &img, int thr);

// extracts the landmarks defining the left pupil
std::vector<Point> left_pupil(const std::vector<Point2f> &landmarks);

// extracts the landmarks defining the right pupil
std::vector<Point> right_pupil(const std::vector<Point2f> &landmarks);

// extracts the landmarks defining the face skin
std::vector<Point> face_oval(const std::vector<Point2f> &landmarks);
