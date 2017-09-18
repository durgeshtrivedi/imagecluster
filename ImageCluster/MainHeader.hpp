//
//  MainHeader.hpp
//  OpencvIOS
//
//  Created by Durgesh Trivedi on 21/06/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#ifndef MainHeader_h
#define MainHeader_h

// This code is specific to read the Current Dir on differenct system without it on xcode not able to read CD
#ifdef _WIN32
#include "direct.h"
#define PATH_SEP '\\'
#define GETCWD _getcwd
#define CHDIR _chdir
#else
#include "unistd.h"
#define PATH_SEP '/'
#define GETCWD getcwd
#define CHDIR chdir
#endif
//

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

#define  CURRENTDIR  GetCurrentWorkingDirectory();
#define  pathlandmarkdetector "/model/shape_predictor_68_face_landmarks.dat";
#define  pathRESNETModel  "/model/dlib_face_recognition_resnet_model_v1.dat";


using namespace cv;
using namespace std;
using std::cout;
using std::endl;
using std::string;

enum CLUSTER 
{
    CREATE_DESCRIPTOR       = -2,
    NEW_FACE                = -1,
};

// Code specific to read Current Dir
string GetExecutableDirectory(const char* argv0);
bool ChangeDirectory(const char* dir);
string GetCurrentWorkingDirectory();

inline bool fileExist (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void copyFile(string SRC, string DEST, string filename);
//

void imageCluster ();
#endif /* MainHeader_h */
