//
//  OpenCVHeader.hpp
//  OpencvIOS
//
//  Created by Durgesh Trivedi on 21/06/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#ifndef OpenCVHeader_h
#define OpenCVHeader_h



#include "opencv2/opencv.hpp"
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
// image classfier
#include "opencv2/objdetect.hpp"

#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <vector>

// tracking object
#include <opencv2/tracking.hpp>

#include "opencv2/core.hpp"

#include "opencv2/ml.hpp"
#include <stdio.h>

// Dlib dnn

#include <dlib/dnn.h>
#include <dlib/image_io.h>

// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
#ifdef _WIN32
#include "dirent.h"
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#include <dirent.h>
#else
#error "Not Mac. Find al alternative to dirent"
#endif
#elif __linux__
#include <dirent.h>
#elif __unix__ // all unices not caught above
#include <dirent.h>
#else
#error "Unknown compiler"
#endif

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO_DLIB 1

// You might need to change the value of it between 1.0 and 1.5 based on different example
#define FACE_DOWNSAMPLE_RATIO 1.5
// You might need to change the value of it between 1 and 2 based on different example
#define SKIP_FRAMES 1


#define OPENCV_FACE_RENDER

#define faceWidth 64
#define faceHeight 64
#define PI 3.14159265

#define THRESHOLD 0.5

#define  pathlandmarkdetector "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/models/shape_predictor_68_face_landmarks.dat";
#define  pathRESNETModel "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/models/dlib_face_recognition_resnet_model_v1.dat";

using namespace cv;

//using namespace cv::ml;
//using  dlib :: frontal_face_detector;



#endif /* OpenCVHeader_h */
