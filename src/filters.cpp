/*
take an image
step 1: gaussian kernel
step 2: sobel filters using gradient calculation
step 3: non-maximum suppression
step 4: double threshold
step 5: hysteresis
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/filters.hpp"

using namespace cv;
#define kernel_size 3

int kernel[3][3] = {1, 2, 1,
                2, 4, 2,
                1, 2, 1};

int Kx[3][3] = {-1, 0, 1,
            -2, 0, 2,
            -1, 0, 1};

int Ky[3][3] = {1, 2, 1,
            0, 0, 0,
            -1, -2, -1};



void gaussianBlur(Frame &orig, Frame &res) {

    int width = orig->width;
    int height = orig->height;
    
    int pixel_value = 0;
    int res_value = 0;
    int row, nrow = 0;
    int col, ncol = 0;
    int weightedSum, sum = 0;

    // launch a kernel here
    for (int index = 0; index < width*height; index ++) {
        
        row = index / height;
        col = index % width;
        weightedSum, sum = 0;
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                nrow = row + i;
                nrow = col + j;
                if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
                    pixel_value = orig->data[index];
                    weightedSum += pixel_value * kernel[i+1][j+1];
                    sum += kernel[i+1][j+1];
                }
            }
        }
        res->data[index] = (unsigned char)weightedSum/sum;
    }

    res->height = height;
    res->width = width;

    return res;
}


void sobelFilters(Frame &orig, Frame &gradient, Frame &angle) {

    int width = orig->width;
    int height = orig->height;
    int row, nrow = 0;
    int col, ncol = 0;
    int sumX, sumY = 0;

    for (int index = 0; index < width*height; index ++) {
        
        row = index / height;
        col = index % width;
        sumX, sumY = 0;
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                nRow = row + j;
                nCol = col + i;
                if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
                    nIndex = nRow * width + nCol;
                    sumX += orig->data[index] * Kx[i+1][j+1];
                    sumY += orig->data[index] * Kx[i+1][j+1];
                }
            }
        }

        gradient->width = orig->width;
        gradient->height = orig->height;
        angle->width = orig->width;
        angle->height = orig->height;

        gradient->data[index] = (unsigned char)sqrt(sumX * sumX + sumY * sumY);
        angle->data[index] = (unsigned char)(atan2(sumY, sumX) * 180) / 3.1415; // pi macro?? 
        

    }
}

void nonMaximumSuppression(Frame &gradient, Frame &angle, Frame &res);

void doubleThreshold(Frame &orig, Frame &res, int lo, int hi);

void hysteresis(Frame &orig, Frame &res, int lo, int hi);