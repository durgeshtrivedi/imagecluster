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


string pathFace = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/faces";
string pathFaceRec = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/FaceRec/trainFaces";
string pathTestFaceRec = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/FaceRec/testFaces/";
string pathEigenFaceRec = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/face_model_eigen.yml";
string pathFisherFaceRec = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/face_model_fisher.yml";
string pathLBHPFaceRec = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/face_model_lbhp.yml";
string pathDescriptorsCSV= "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/descriptors.csv";

string pathVaibhav = "/Users/durgeshtrivedi/durgesh/OpenCV/LearnOpenCV/IOSWithOpencv/OpencvIOS/opencv/data/images/faces/vaibhaw_query.jpg";

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

#endif /* ImageCluster_hpp */
