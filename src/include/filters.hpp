#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "global.hpp"

/*
take an image
step 1: gaussian kernel
step 2: sobel filters using gradient calculation
step 3: non-maximum suppression
step 4: double threshold
step 5: hysteresis
*/

struct Frame {
    int width;
    int height;
    unsigned char *data;
};

void gaussianBlur(Frame &orig, Frame &res);

void sobelFilters(Frame &orig, Frame &gradient, Frame &angle);

void nonMaximumSuppression(Frame &gradient, Frame &angle, Frame &res);

void doubleThreshold(Frame &orig, Frame &res, int lo, int hi);

void hysteresis(Frame &orig, Frame &res, int lo, int hi);