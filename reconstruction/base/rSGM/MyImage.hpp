// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#include <string>
#include "StereoCommon.h"
#include <cstring>
#include <sstream>

uint32 readNumber(uint8* p, uint32& index)
{
    uint32 start = index;
    while (p[index] != ' ' && p[index] != '\n')
    {
        index++;
    }
    std::string str;
    str.append((char*)p+start, index - start);
    uint32 result = atoi(str.c_str());
    index++;
    return result;
}

template <typename T>
void readPGM(MyImage<T>& img, const char* filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (file.is_open())
    {
        // get length of file:
        file.seekg(0, file.end);
        uint32 length = (uint32)file.tellg();
        file.seekg(0, file.beg);
    
        // read whole file
        uint8* p = (uint8*)_mm_malloc(length, 16);

        file.read((char*)p, length);
        file.close();

        // parse header
        if (p[0] != 'P' || p[1] != '5') {
            std::cerr << "wrong magic number, only P5 PGM files supported" << filename << std::endl;
            _mm_free(p);
            return;
        }
        uint32 index = 3;
        uint32 width = readNumber(p, index);
        uint32 height = readNumber(p, index);
        uint32 maxValue = readNumber(p, index);

        // size checks
        if (sizeof(T) == 1)
        {
            if (maxValue != 255) {
                std::cerr << "bit depth of pgm does not match image type" << filename << std::endl;
                _mm_free(p);
                return;
            }
        }
        if (sizeof(T) == 2)
        {
            if (maxValue == 255) {
                std::cerr << "bit depth of pgm does not match image type" << filename << std::endl;
                _mm_free(p);
                return;
            }
        }
        // check values
        if (length - index - width*height*sizeof(T) != 0)
        {
            std::cerr << "error in image parsing, header does not match file length" << filename << std::endl;
            _mm_free(p);
            return;
        }
        // copy values
        T* data = (T*)_mm_malloc(width*height*sizeof(T), 16);

        memcpy(data, p + index, width*height*sizeof(T));

        img.setAttributes(width, height, data);
        
        _mm_free(p);
    }
    else
    {
        std::cerr << "could not open file to read "<< filename << std::endl;
    }
}

template <typename T>
void writePGM(MyImage<T>& img, const char* filename, bool verbose)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open())
    {
        // write header
        file << "P5\n";

        std::ostringstream s;
        s << img.getWidth()<<" "<<img.getHeight()<<"\n";
        if (sizeof(T) == 1)
            s << "255\n";
        else
        s << "65535\n";
        file << s.str();
        file.write((char*)img.getData(), img.getWidth()*img.getHeight()*sizeof(T));
        file.close();

        if (verbose)
        {
            std::cout << "write to file "<< filename << std::endl;
        }
    }
    else std::cout << "Unable to open file "<<filename << std::endl;
}