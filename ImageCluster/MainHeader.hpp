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

#include <sys/stat.h>
#define faceWidth 64
#define faceHeight 64
#define PI 3.14159265

#define THRESHOLD 0.5

#define  CURRENTDIR  GetCurrentWorkingDirectory();
#define  pathlandmarkdetector "/model/shape_predictor_68_face_landmarks.dat";
#define  pathRESNETModel  "/model/dlib_face_recognition_resnet_model_v1.dat";

#define STRTOLOWER(x) std::transform (x.begin(), x.end(), x.begin(), ::tolower)
#define STRTOUPPER(x) std::transform (x.begin(), x.end(), x.begin(), ::toupper)
#define STRTOUCFIRST(x) std::transform (x.begin(), x.begin()+1, x.begin(),  ::toupper); std::transform (x.begin()+1, x.end(),   x.begin()+1,::tolower)


using namespace cv;
using namespace std;
using std::cout;
using std::endl;
using std::string;

#define  CREATE_DESCRIPTOR  "DESCRIPTOR"
#define  NEW_FACE "NEWFACE"

  //To increase the speed of faceDetector we need to resize the iamge, so With 576 size the result is good for images with single face and also for images with 4 5 face up to some extent with size 300 for single face the speed increase 3X but produce bad result for 2 3 faces
#define RESIZE_HEIGHT 576
#define FACE_DOWNSAMPLE_RATIO_DLIB 4

enum OPTIONS
{
    OPTION_1_CLUSTER_ALL_FACES          = 100,
    OPTION_2_READ_FIRST_FOLDERS         = 101,
    OPTION_2_READ_FIRST_THAN_CLUSTER    = 102,
    OPTION_EXIT                         = 0
};

// Code specific to read Current Dir
string GetExecutableDirectory(const char* argv0);

bool ChangeDirectory(const char* dir);

/**
 Function to Get the current working DIR on a system
 */
string GetCurrentWorkingDirectory();

/**
 Function to check the file exist at give path
 param string : file path
 */
inline bool fileExist (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void imageCluster ();

/**
 It will check if the given path is a dir or not
 param const char* : path DIR path
 */
bool is_dir(const char* path);

/**
 It will check if the given path is a file or not
 param const char* : path file path
 */
bool is_file(const char* path);

/**
 Copy function to copy a file from source to destination location
 param string : SRC file path
 param string : DEST file path
 param string : filename
 */
void copyFile(string SRC, string DEST, string filename);
//
#endif /* MainHeader_h */
