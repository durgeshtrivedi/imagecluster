//
//  ImageCluster.cpp
//  ImageCluster
//
//  Created by Durgesh Trivedi on 11/09/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#include "ImageCluster.hpp"

void imageCluster () {
    int index;
    cout << "You want to train enter 1 want to test the train data enter 2 : ";
    cin >> index;
    switch (index) {
        case 1 :
            cout << "Reading all the images to enroll will take some time Keep patience ... <<" << endl;
            enRollDlibFaceRec();
            break;
        case 2 :
            cout << "Showing will take some time based on how large the data ...  <<" << endl;
            //readImages();
            testDlibFaceRecImage();
            break;
        default :
            cout << "Input the correct value" << endl;
            imageCluster ();
            break;
    }
}

// ----------------------------------------------------------------------------------------

template<typename T>
void printVector(std::vector<T>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << i << " " << vec[i] << "; ";
    }
    cout << endl;
}



void enRollDlibFaceRec() {
    cout << "It will take some time to read the dlib model keep patience " << endl;
    // Initialize face detector, facial landmarks detector and face recognizer
    string currectDir  = CURRENTDIR ;
    String predictorPath, faceRecognitionModelPath;
    predictorPath =  currectDir + pathlandmarkdetector;
    
    faceRecognitionModelPath =   currectDir + pathRESNETModel;
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;
    
    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person
    string faceDatasetFolder = currectDir + pathFace;
    std::vector<string> subfolders, fileNames, symlinkNames;
    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);
    
    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<string> names;
    std::vector<int> labels;
    std::map<int, string> labelNameMap;
    // add -1 integer label for un-enrolled persons
    names.push_back("unknown");
    labels.push_back(-1);
    
    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<string> imagePaths;
    std::vector<int> imageLabels;
    
    // variable to hold any subfolders within person subFolders
    std::vector<string> folderNames;
    // iterate over all subFolders within faces folder
    for (int i = 0; i < subfolders.size(); i++) {
        string personFolderName = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = personFolderName.find_last_of("/\\");
        string name = personFolderName.substr(found+1);
        // assign integer label to person subFolder
        int label = i;
        // add person name and label to vectors
        names.push_back(name);
        labels.push_back(label);
        // add (integer label, person name) pair to map
        labelNameMap[label] = name;
        
        // read imagePaths from each person subFolder
        // clear vectors
        folderNames.clear();
        fileNames.clear();
        symlinkNames.clear();
        // folderNames and symlinkNames are useless here
        // as we are only looking for files here
        // read all files present in subFolder
        listdir(subfolders[i], folderNames, fileNames, symlinkNames);
        // filter only jpg files
        filterFiles(subfolders[i], fileNames, imagePaths, "jpg");
        // add label i for all ajpg files found in subFolder
        for (int j = 0; j < imagePaths.size(); j++) {
            imageLabels.push_back(i);
        }
    }
    
    // process training data
    // We will store face descriptors in vector faceDescriptors
    // and their corresponding labels in vector faceLabels
    std::vector<matrix<float,0,1>> faceDescriptors;
    // std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
    std::vector<int> faceLabels;
    
    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++) {
        string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];
        
        cout << "processing: " << imagePath << endl;
        
        // read image using OpenCV
        Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);
        
        // convert image from BGR to RGB
        // because Dlib used RGB format
        Mat imRGB;
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
        
        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));
        
        // detect faces in image
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        // Now process each face we found
        for (int j = 0; j < faceRects.size(); j++) {
            
            // Find facial landmarks for each detected face
            full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);
            
            // object to hold preProcessed face rectangle cropped from image
            matrix<rgb_pixel> face_chip;
            
            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);
            
            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            matrix<float,0,1> faceDescriptor = net(face_chip);
            
            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels
            faceDescriptors.push_back(faceDescriptor);
            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            faceLabels.push_back(imageLabel);
        }
    }
    
    cout << "number of face descriptors " << faceDescriptors.size() << endl;
    cout << "number of face labels " << faceLabels.size() << endl;
    
    // write label name map to disk
    const string labelNameFile = textFilePath;
    ofstream of;
    of.open (labelNameFile);
    for (int m = 0; m < names.size(); m++) {
        of << names[m];
        of << ";";
        of << labels[m];
        of << "\n";
    }
    of.close();
    
    // write face labels and descriptor to disk
    // each row of file descriptors.csv has:
    // 1st element as face label and
    // rest 128 as descriptor values
    const string descriptorsPath = currectDir + pathDescriptorsCSV;
    ofstream ofs;
    ofs.open(descriptorsPath);
    // write descriptors
    for (int m = 0; m < faceDescriptors.size(); m++) {
        matrix<float,0,1> faceDescriptor = faceDescriptors[m];
        std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());
        // cout << "Label " << faceLabels[m] << endl;
        ofs << faceLabels[m];
        ofs << ";";
        for (int n = 0; n < faceDescriptorVec.size(); n++) {
            ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];
            // cout << n << " " << faceDescriptorVec[n] << endl;
            if ( n == (faceDescriptorVec.size() - 1)) {
                ofs << "\n";  // add ; if not the last element of descriptor
            } else {
                ofs << ";";  // add newline character if last element of descriptor
            }
        }
    }
    ofs.close();
    
    
    return ;
}

void readImages() {
    string currectDir  = CURRENTDIR;
    String predictorPath, faceRecognitionModelPath;
    predictorPath =  currectDir + pathlandmarkdetector;
    
    faceRecognitionModelPath =  currectDir + pathRESNETModel;
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;
    
    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person
    string faceDatasetFolder = currectDir + pathFace;
    std::vector<string> subfolders, fileNames, symlinkNames;
    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);
    
    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<string> names;
    std::vector<int> labels;
    std::map<int, string> labelNameMap;
    // add -1 integer label for un-enrolled persons
    names.push_back("unknown");
    labels.push_back(-1);
    
    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<string> imagePaths;
    std::vector<int> imageLabels;
    
    // variable to hold any subfolders within person subFolders
    std::vector<string> folderNames;
    // iterate over all subFolders within faces folder
    for (int i = 0; i < subfolders.size(); i++) {
        string personFolderName = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = personFolderName.find_last_of("/\\");
        string name = personFolderName.substr(found+1);
        // assign integer label to person subFolder
        int label = i;
        // add person name and label to vectors
        names.push_back(name);
        labels.push_back(label);
        // add (integer label, person name) pair to map
        labelNameMap[label] = name;
        
        // read imagePaths from each person subFolder
        // clear vectors
        folderNames.clear();
        fileNames.clear();
        symlinkNames.clear();
        // folderNames and symlinkNames are useless here
        // as we are only looking for files here
        // read all files present in subFolder
        listdir(subfolders[i], folderNames, fileNames, symlinkNames);
        // filter only jpg files
        filterFiles(subfolders[i], fileNames, imagePaths, "jpg");
        // add label i for all ajpg files found in subFolder
        for (int j = 0; j < imagePaths.size(); j++) {
            imageLabels.push_back(i);
        }
    }
   separateImage(faceDetector, landmarkDetector, net, imagePaths, imageLabels, folderNames);
}

void separateImage(frontal_face_detector faceDetector,
                   shape_predictor landmarkDetector,
                   anet_type net,
                   std::vector<string> imagePaths,
                   std::vector<int> imageLabels,
                   // variable to hold any subfolders within person subFolders
                   std::vector<string> folderNames) {
    // Initialize face detector, facial landmarks detector and face recognizer
    
    
    // read names, labels and labels-name-mapping from file
    string currentDir = CURRENTDIR;
    std::map<int, string> labelNameMap;
    std::vector<string> names;
    std::vector<int> labels;
    const string labelNameFile = currentDir + textFilePath;
    readLabelNameMap(labelNameFile, names, labels, labelNameMap);
    
    // read descriptors of enrolled faces from file
    const string faceDescriptorFile =  currentDir + pathDescriptorsCSV;
    std::vector<int> faceLabels;
    std::vector<matrix<float,0,1>> faceDescriptors;
    readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);
    
    // read query image
    // string imagePath;
    
    //imagePath = currectDir + pathVaibhav;
    
    // iterate over images
    for (int index = 0; index < imagePaths.size(); index++) {
        string imagePath = imagePaths[index];
        int imageLabel = imageLabels[index];
        
        cout << "processing: " << imagePath << endl;
        
        Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);
        
        if (im.empty()){
            exit(0);
        }
        
        // convert image from BGR to RGB
        // because Dlib used RGB format
        Mat imRGB = im.clone();
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
        
        
        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));
        
        // detect faces in image
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        // Now process each face we found
        for (int i = 0; i < faceRects.size(); i++) {
            
            // Find facial landmarks for each detected face
            full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i]);
            
            // object to hold preProcessed face rectangle cropped from image
            matrix<rgb_pixel> face_chip;
            
            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(imDlib, get_face_chip_details(landmarks,150,0.25), face_chip);
            
            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            matrix<float,0,1> faceDescriptorQuery = net(face_chip);
            
            // Find closest face enrolled to face found in frame
            int label;
            float minDistance;
            nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
            // Name of recognized person from map
            string name = labelNameMap[label];
            
            // Draw a rectangle for detected face
            /* Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
             Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
             cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);
             
             // Draw circle for face recognition
             Point2d center = Point((faceRects[i].left() + faceRects[i].right())/2.0,
             (faceRects[i].top() + faceRects[i].bottom())/2.0 );
             int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top())/2.0);
             cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);
             
             // Write text on image specifying identified person and minimum distance
             stringstream stream;
             stream << name << " ";
             stream << fixed << setprecision(4) << minDistance;
             string text = stream.str(); // name + " " + std::to_string(minDistance);
             cv::putText(im, text, p1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2); */
            cout.precision( numeric_limits<double>::digits + 1);
            cout << "THis is the minimum distance for the current Image: " << fixed << minDistance  << endl;
            if (label != -1) {
                std::size_t pos = imagePath.find_last_of("/");
                string value = imagePath.substr(pos);
                string sorceDir = imagePath.substr(0, pos);
                string destDir = currentDir +  resultPath + "Hello";
                copyFile(sorceDir, destDir, value);
            }
        }
    }
    // Show result
    //cv::imshow("webcam", im);
    int k = cv::waitKey(0);
    
    cv::destroyAllWindows();
}




void testDlibFaceRecImage() {
    // Initialize face detector, facial landmarks detector and face recognizer
    String predictorPath, faceRecognitionModelPath;
    string currectDir  = CURRENTDIR
    predictorPath = currectDir + pathlandmarkdetector;
    faceRecognitionModelPath = currectDir + pathRESNETModel;
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;
    
    // read names, labels and labels-name-mapping from file
    std::map<int, string> labelNameMap;
    std::vector<string> names;
    std::vector<int> labels;
    const string labelNameFile = currectDir + textFilePath;
    readLabelNameMap(labelNameFile, names, labels, labelNameMap);
    
    // read descriptors of enrolled faces from file
    const string faceDescriptorFile = currectDir + pathDescriptorsCSV;
    std::vector<int> faceLabels;
    std::vector<matrix<float,0,1>> faceDescriptors;
    readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);
    
    // read query image
    string imagePath;
    
    imagePath = currectDir + pathVaibhav;
    
    Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (im.empty()){
        exit(0);
    }
    
    // convert image from BGR to RGB
    // because Dlib used RGB format
    Mat imRGB = im.clone();
    cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
    
    
    // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
    // Dlib's dnn module doesn't accept Dlib's cv_image template
    dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));
    
    // detect faces in image
    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    // Now process each face we found
    for (int i = 0; i < faceRects.size(); i++) {
        
        // Find facial landmarks for each detected face
        full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i]);
        
        // object to hold preProcessed face rectangle cropped from image
        matrix<rgb_pixel> face_chip;
        
        // original face rectangle is warped to 150x150 patch.
        // Same pre-processing was also performed during training.
        extract_image_chip(imDlib, get_face_chip_details(landmarks,150,0.25), face_chip);
        
        // Compute face descriptor using neural network defined in Dlib.
        // It is a 128D vector that describes the face in img identified by shape.
        matrix<float,0,1> faceDescriptorQuery = net(face_chip);
        
        // Find closest face enrolled to face found in frame
        int label;
        float minDistance;
        nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
        // Name of recognized person from map
        string name = labelNameMap[label];
        
        // Draw a rectangle for detected face
        Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
        Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
        cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);
        
        // Draw circle for face recognition
        Point2d center = Point((faceRects[i].left() + faceRects[i].right())/2.0,
                               (faceRects[i].top() + faceRects[i].bottom())/2.0 );
        int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top())/2.0);
        cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);
        
        // Write text on image specifying identified person and minimum distance
        stringstream stream;
        stream << name << " ";
        stream << fixed << setprecision(4) << minDistance;
        string text = stream.str(); // name + " " + std::to_string(minDistance);
        cv::putText(im, text, p1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
    }
    
    // Show result
    cv::imshow("webcam", im);
    int k = cv::waitKey(0);
    
    cv::destroyAllWindows();
}
