//
//  ImageCluster.hpp
//  ImageCluster
//
//  Created by Durgesh Trivedi on 11/09/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#ifndef ImageCluster_hpp
#define ImageCluster_hpp

#include "FaceBlendCommon.hpp"


string pathFace =  "/data/faces";
string resultPath =  "/result/";
string pathDescriptorsCSV =  "/result/descriptors.csv";
string pathVaibhav = "/data/faces/vaibhaw_query.jpg";
string textFilePath =  "/result/label_name.txt";


using namespace dlib;
// ----------------------------------------------------------------------------------------
// The next bit of code defines a ResNet network. It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
alevel0<
alevel1<
alevel2<
alevel3<
alevel4<
max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
input_rgb_image_sized<150>
>>>>>>>>>>>>;

void imageCluster ();
void userFolderDir();
bool readRootDir();

void allImageDirector(OPTIONS option);
void clusterFaces(OPTIONS options);
void clusterAllFaces(string label,
                     string imagePath,
                     matrix<float,0,1> &faceDescriptorQuery,
                     std::vector<matrix<float,0,1>> &faceDescriptors,
                     std::vector<string> &faceLabels,
                     char  &alphabet);
void clusterUserFaces(string label,
                      string imagePath,
                      matrix<float,0,1> &faceDescriptorQuery,
                      std::vector<matrix<float,0,1>> &faceDescriptors,
                      std::vector<string> &faceLabels);
string faceMatch(string descriptorDir, matrix<float,0,1> &faceDescriptorQuery, std::vector<string> &faceLabels);

void writeDescriptors(std::vector<string> &faceLabels, std::vector<matrix<float,0,1>> &faceDescriptors);

void readFolder(std::vector<string> &imagePaths,
                std::vector<string> &imageLabels);
void moveSelectedFaces(string label, string imagePath);
void saveFile(string label, string imagePath);
void saveDescriptor(string imagePath,
                             matrix<float,0,1> &faceDescriptorQuery,
                             std::vector<matrix<float,0,1>> &faceDescriptors,
                             std::vector<string> &faceLabels,
                              string  &alphabet);
void separateImage(frontal_face_detector faceDetector,
                   shape_predictor landmarkDetector,
                   anet_type net,
                   std::vector<string> imagePaths,
                   std::vector<int> imageLabels);
#endif /* ImageCluster_hpp */
