// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#pragma once

#include "StereoCommon.h"

template <typename T>
class MyImage {
public:
    MyImage() :m_data(NULL)
        , m_width(0)
        , m_height(0)
    {};

    MyImage(uint32 width, uint32 height) :
        m_width(width)
        , m_height(height)
    {
        m_data = (T*)_mm_malloc(width*height*sizeof(T), 16);
    };

    MyImage(uint32 width, uint32 height, T* data) :m_data(data)
        , m_width(width)
        , m_height(height)
    {};

    ~MyImage()
    {
        if (m_data != NULL)
            _mm_free(m_data);
    }

    void setAttributes(uint32 width, uint32 height, T* data)
    {
        m_width = width;
        m_height = height;
        m_data = data;
    }

    uint32 getWidth()
    {
        return m_width;
    }

    uint32 getHeight()
    {
        return m_height;
    }

    T* getData()
    {
        return m_data;
    }

private:
    T* m_data;
    uint32 m_width;
    uint32 m_height;
};

template <typename T>
void readPGM(MyImage<T>& img, const char* filename);
template <typename T>
void writePGM(MyImage<T>& img, const char* filename, bool verbose);

#include "MyImage.hpp"
