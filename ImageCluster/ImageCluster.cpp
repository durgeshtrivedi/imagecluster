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
    cout << "You want to Segregate all your images based on faces. Please enter 1 :" << endl;
    cin >> index;
    while(std::cin.fail()) {
        cout << "Input the correct value it should be integer only." << endl;
        std::cin.clear();
        std::cin.ignore(256,'\n');
        imageCluster ();
    }
    switch (index) {
        case 1 :
            cout << "Reading all the images to enroll will take some time Keep patience ... <<" << endl;
            enRollDlibFaceRec();
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
    string currentDir  = CURRENTDIR ;
    String predictorPath, faceRecognitionModelPath;
    predictorPath =  currentDir + pathlandmarkdetector;
    
    faceRecognitionModelPath =   currentDir + pathRESNETModel;
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;
    std::vector<string> imagePaths;
    std::vector<int> imageLabels;
    
    readFolder(imagePaths, imageLabels);
    
    char alphabet = 'A';
    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++) {
        string imagePath = imagePaths[i];
        // process training data
        // We will store face descriptors in vector faceDescriptors
        // and their corresponding labels in vector faceLabels
        std::vector<matrix<float,0,1>> faceDescriptors;
        std::vector<int> faceLabels;
        
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
            matrix<float,0,1> faceDescriptorQuery = net(face_chip);
            
            // Match the face already in the model
            int label = faceMatch(faceDescriptorQuery, faceLabels);
            
            switch (label) {
                case CREATE_DESCRIPTOR:
                    // If there is no descriptor file save model for first time
                    saveDescriptor(imagePath, faceDescriptorQuery, faceDescriptors, faceLabels, alphabet);
                    break;
                    // update model with new Face Match
                case NEW_FACE:
                    // Change the folder name before saving it as new face nmatch
                    alphabet++;
                    saveDescriptor(imagePath, faceDescriptorQuery, faceDescriptors, faceLabels, alphabet);
                    break;
                default:
                {
                    // update model with new learning data
                    // If model match the face copy the file inside the label path
                    std::size_t pos = imagePath.find_last_of("/");
                    string value = imagePath.substr(pos);
                    string sorceDir = imagePath.substr(0, pos);
                    string destDir = currentDir +  resultPath + (char)label;
                    copyFile(sorceDir, destDir, value);
                }
                    break;
            }
        }
    }
    return ;
}

void saveDescriptor(string imagePath,
                    matrix<float,0,1> &faceDescriptorQuery,
                    std::vector<matrix<float,0,1>> &faceDescriptors,
                    std::vector<int> &faceLabels,
                    char  &alphabet) {
    
    string currentDir  = CURRENTDIR ;
    std::size_t pos = imagePath.find_last_of("/");
    string value = imagePath.substr(pos);
    string sorceDir = imagePath.substr(0, pos);
    string destDir = currentDir +  resultPath + (char)alphabet;
    copyFile(sorceDir, destDir, value);
    
    // if label is -1 clear the faceLabels, faceDescriptors
    faceLabels.clear();
    faceDescriptors.clear();
    // add face descriptor and label for this face to
    // vectors faceDescriptors and faceLabels
    faceDescriptors.push_back(faceDescriptorQuery);
    // add label for this face to vector containing labels corresponding to
    // vector containing face descriptors
    faceLabels.push_back(alphabet);
    
    // save the label and descriptor to the file
    writeDescriptors(faceLabels,faceDescriptors);
}
void writeDescriptors(std::vector<int> &faceLabels, std::vector<matrix<float,0,1>> &faceDescriptors) {
    cout << "number of face descriptors " << faceDescriptors.size() << endl;
    cout << "number of face labels " << faceLabels.size() << endl;
    
    string currentDir = CURRENTDIR;
    
    
    // write face labels and descriptor to disk
    // each row of file descriptors.csv has:
    // 1st element as face label and
    // rest 128 as descriptor values
    const string descriptorsPath = currentDir + pathDescriptorsCSV;
    
    std::fstream ofs(descriptorsPath, std::ios::out | std::ios::app);
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
    
}

int faceMatch(matrix<float,0,1> &faceDescriptorQuery, std::vector<int> &faceLabels) {
    
    string currentDir = CURRENTDIR;
    // read descriptors of enrolled faces from file
    const string faceDescriptorFile =  currentDir + pathDescriptorsCSV;
    
    if (fileExist(faceDescriptorFile) == false) {
        // The pathDescriptor file doesn't exist means file not yet created and it is first face
        return -2;
    }
    
    std::vector<matrix<float,0,1>> faceDescriptors;
    readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);
    
    // Find closest face enrolled to face found in frame
    int label;
    float minDistance;
    nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
    return label;
}
void readFolder(std::vector<string> &imagePaths,
                std::vector<int> &imageLabels) {
    
    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person
    string currentDir = CURRENTDIR ;
    string faceDatasetFolder = currentDir + pathFace;
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
        std::vector<string> extensions{"jpg","png"};
        filterFiles(subfolders[i], fileNames, imagePaths, extensions);
        // add label i for all ajpg files found in subFolder
        for (int j = 0; j < imagePaths.size(); j++) {
            imageLabels.push_back(i);
        }
    }
}
