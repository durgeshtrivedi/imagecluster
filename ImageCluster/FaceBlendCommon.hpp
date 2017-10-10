
#ifndef FaceBlendCommon_HPP_
#define FaceBlendCommon_HPP_

#include "MainHeader.hpp"


#ifndef M_PI
  #define M_PI 3.14159
#endif


using namespace dlib;

/**
 Get the cropped region from a face
 param Mat                        : image  image input to cropped
 param vector<Point2f>            : landmarks face landmarks point
 param Rect &                     : selectedRegion  rectangle region for face point
 */
Mat getCroppedFaceRegion(Mat image, std::vector<Point2f> landmarks, cv::Rect &selectedRegion);



/**
 Read descriptor saved on disk
 param string                     : filename
 param vector<string>&            : faceLabels  facelabel array for all the faces
 param vector<matrix<float,0,1>>& : faceDescriptors  faceDescriptors to mach a face
 param char                       : separator
 */
void readDescriptors(const string& filename, std::vector<string>& faceLabels, std::vector<matrix<float,0,1>>& faceDescriptors, char separator = ';');


/**
 Find nearest face descriptor from vector of enrolled faceDescriptor
 to a query face descriptor
 param matrix<float, 0, 1>&                         : faceDescriptorQuery faceDescriptor which need to match for nearest  neighbour
 param vector<dlib::matrix<float, 0, 1>>&           : faceDescriptor
 param string &                                     : lable for match face
 */
void nearestNeighbor(dlib::matrix<float, 0, 1>& faceDescriptorQuery,
                     std::vector<dlib::matrix<float, 0, 1>>& faceDescriptors,
                     std::vector<string>& faceLabels, string &label);

/**
 Filter files from given files (jpg,png)
 param string                     : dirPath path from images to read
 param vector<string>&            : fileNames  fileNames to be filder in the dir
 param std::vector<string>&       : filteredFilePaths  path for file which are filtered
 param vector<string>             : file extension to be filter (jpg,png)
 */
void filterFiles(string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, std::vector<string> extensions);

void listdir(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames);

#endif 
