# README #

If you try to run the project and you are faceing issue just look into  PROJECT SETUP section else follow RUN section  

### RUN THE PROJECT ###
If you just want to run it after setup the ### version_0.2 ### has 2 features and which can be run from command line only right now.
So make sure which version tag you are using to execute it.

Once run it will show options on command line

    You have mutiple options to chose from.
       1. You want to Segregate all your images based on faces.
    
       2. If you want to segregate only images with particular person init from a group of images.
         then you need to provide 1 or 2 images of that person in a folder with the name you would like 
         to have so that system can learn which images only you want, you can also keep mutiple folder
         with different person in it to get all the images of those person only in those folder.
       3. 0 to exit the app.

Based on the option you chose you get more option.

It is more important you keep the folder path for images as mentioned in options else app will not behave as intended, its a limitation.
The drag and drop for folder path is not working, its again a limitation for current version, so you need to type it or copy paste it.

### PROJECT SETUP 
This project can only be run using Xcode on MAC.It is having default setting considering OpencCV and Dlib is installed on the system with specific path else 
you need to edit the Opencv and dlib Header/ library path in build setting for the project. Which you definately need it to run the project .

### OpenCV  and Dlib header search path in project setting ###
The OpenCv 3.2.0 and dlib-19.4 version I have used for this project, higher version should also work for both

The header search path for dlib and opencv should be like this 

First you need to install the Opencv on your sytem for that. Also you need build dlib file .

/usr/local/Cellar/opencv3/3.2.0 

Follow these links to setup Opencv path 

http://charliegerard.github.io/blog/Installing-OpenCV/ 

https://blogs.wcode.org/2014/11/howto-setup-xcode-6-1-to-work-with-opencv-libraries/ 

You should have the dlib-19.4 header in a directory and give its path  like below 

/Users/durgeshtrivedi/durgesh/OpenCV/dlib-19.4

### OpenCv and Dlib Library search path in project setting ###
The library search path for Opencv is 

/usr/local/Cellar/opencv3/3.2.0

The library search path for Dlib 

You need to keep the libdlib.a file in lib folder inside the project dir. Which is actually part of project you no need to do any thing her actually
$(PROJECT_DIR)/ImageCluster/lib


### Dlib in build phase ##

 You need to add the libdlib.a  and Accelerate.framwork in link binary
 
 You again no need to do any thing over here 
 
 ### OpenCv Other linker flags ##
 
 add this flags 
 
 -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab
 
 again you no need to add it is already there
 
 ### dlib preprocess macros 
 
To build the  project properly you need to add some preprocess macro for dlib else it will not build 

These 2 links will help to understand it. again you no need to do it now it is part of project .
Debug 

NDEBUG
DDLIB_PNG_SUPPORT
DDLIB_JPEG_SUPPORT
DLIB_NO_GUI_SUPPORT

Release
NDEBUG
DDLIB_USE_LAPACK
DDLIB_USE_BLAS

https://github.com/zweigraf/face-landmarking-ios/issues/7 

https://github.com/RedHandTech/Object_Detector  

### Notes ## 
So to make this project build properly you only need to properly give path for headers for opencv and dlib .
Rest of the setting are already part of the project and they should work all the info in just for understanding if any things not work. 

I have added the ziped dlib-19.4 file in extra folder in project dir you can unzip and use the path in headers for dlib.Thats should work I have not tried this way 
because I build dlib from source, so I think this way will work. But you must install OpenCV to run the project .