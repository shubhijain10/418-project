#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

// struct Frame {
//     int width;
//     int height;
//     unsigned char *data;
// };

// __global__ void gaussianBlur(unsigned char* orig, unsigned char* res, int width, int height);
// void sobelFilters(Frame *orig, Frame *gradient, Frame *angle);
// int nonMaximumSuppression(Frame *gradient, Frame *angle, Frame *res);
// void hysteresis(Frame *orig, Frame *res, int maxGradient);

void cudaCanny(unsigned char* inImage, int width, int height, unsigned char* outImage);


#endif