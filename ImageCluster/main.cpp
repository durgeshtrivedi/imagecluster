//
//  main.cpp
//  ImageCluster
//
//  Created by Durgesh Trivedi on 11/09/17.
//  Copyright © 2017 durgesh. All rights reserved.
//

#include "MainHeader.hpp"
#include <sys/fcntl.h>

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


void copyFile(string srcDirPath, string destDirPath, string fileName)
{


    string command = "mkdir -p " + destDirPath;
    //cerr << endl << "Command = " <<  command.c_str() << endl << endl;
    system(command.c_str());


    DIR* pnWriteDir = NULL;    /*DIR Pointer to open Dir*/
    pnWriteDir = opendir(destDirPath.c_str());
    
    if (!pnWriteDir)
        cerr << endl << "ERROR! Write Directory can not be open" << endl;
    
    DIR* pnReadDir = NULL;    /*DIR Pointer to open Dir*/
    pnReadDir = opendir(srcDirPath.c_str());
    
    if (!pnReadDir || !pnWriteDir)
        cerr << endl <<"ERROR! Read or Write Directory can not be open" << endl << endl;
    
    else
    {
        string srcFilePath = srcDirPath + fileName;
        const char * strSrcFileName = srcFilePath.c_str();
        fstream in, out;
        in.open(strSrcFileName, fstream::in|fstream::binary);
        
        if (in.is_open()) {
            //cerr << endl << "Now reading file " << strSrcFileName << endl;
            
            string destFilePath = destDirPath + fileName;
            const char * strDestFileName = destFilePath.c_str();
            out.open(strDestFileName, fstream::out);
            
            char tmp;
            while(in.read(&tmp, 1))
            {
                out.write(&tmp, 1);
            }
            out.close();
            in.close();
        }
        else
            cerr << endl << "ERROR! File Could not be open for reading" << endl;
    }
    closedir(pnReadDir);
    closedir(pnWriteDir);
}
