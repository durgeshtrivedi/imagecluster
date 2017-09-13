//
//  main.cpp
//  ImageCluster
//
//  Created by Durgesh Trivedi on 11/09/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#include "MainHeader.hpp"
int main(int argc, const char * argv[]) {
    imageCluster();
    return 0;
}

// Code specific to read Current Dir
string GetExecutableDirectory(const char* argv0) {
    string path = argv0;
    int path_directory_index = path.find_last_of(PATH_SEP);
    return path.substr(0 , path_directory_index + 1);
}

bool ChangeDirectory(const char* dir) {return CHDIR(dir) == 0;}

string GetCurrentWorkingDirectory() {
    const int BUFSIZE = 4096;
    char buf[BUFSIZE];
    memset(buf , 0 , BUFSIZE);
    GETCWD(buf , BUFSIZE - 1);
    return buf;
}

//
