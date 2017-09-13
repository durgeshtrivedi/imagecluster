//
//  MainHeader.hpp
//  OpencvIOS
//
//  Created by Durgesh Trivedi on 21/06/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#ifndef MainHeader_h
#define MainHeader_h



//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
// image classfier
//#include "opencv2/objdetect.hpp"

//#include <opencv2/core.hpp>
//#include <opencv2/face.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>


#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <map>
#include <stdlib.h>
#include <algorithm>
#include <vector>

// Dlib dnn

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#define faceWidth 64
#define faceHeight 64
#define PI 3.14159265

#define THRESHOLD 0.5

#define  pathlandmarkdetector "./models/shape_predictor_68_face_landmarks.dat";
#define  pathRESNETModel "./models/dlib_face_recognition_resnet_model_v1.dat";

using namespace cv;
using namespace std;
void imageCluster ();
#endif /* MainHeader_h */
