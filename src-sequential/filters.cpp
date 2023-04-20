/*
take an image
step 1: gaussian kernel
step 2: sobel filters using gradient calculation
step 3: non-maximum suppression
step 4: double threshold
step 5: hysteresis
*/
#include <iostream>
#include <cmath>
// #include <opencv2/opencv.hpp>
#include "include/filters.hpp"

using namespace std;
#define kernel_size 3

int kernel3[3][3] = {1, 2, 1,
                2, 4, 2,
                1, 2, 1};

int kernel5[5][5] = {1, 4, 7, 4, 1,
                4, 16, 26, 16, 4,
                7, 26, 41, 26, 7,
                4, 16, 26, 16, 4,
                1, 4, 7, 4, 1};

int Kx[3][3] = {-1, 0, 1,
            -2, 0, 2,
            -1, 0, 1};

int Ky[3][3] = {1, 2, 1,
            0, 0, 0,
            -1, -2, -1};



void gaussianBlur(Frame *orig, Frame *res) {
    cout << "in blurring mode\n";
    int width = orig->width;
    int height = orig->height;
    
    unsigned char pixel_value = 0;
    int res_value = 0;
    int y, ny = 0;
    int x, nx = 0;
    int weightedSum, sum = 0;
    
    for (int index = 0; index < width*height; index ++) {
        y = index / width;
        x = index % height;
        weightedSum = 0;
        sum = 0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                nx = x + i;
                ny = y + j;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    pixel_value = orig->data[ny * width + nx];
                    
                    weightedSum += pixel_value * kernel5[i+2][j+2];
                    //printf("pixel value: %d\n", pixel_value);
                    sum += kernel5[i+2][j+2];
                }
            }
        }
        unsigned char result = weightedSum/sum;
        res->data[index] = (unsigned char)result;
    }
    cout << "done going through the thing\n";
}


void sobelFilters(Frame *orig, Frame *gradient, Frame *angle) {

    int width = orig->width;
    int height = orig->height;
    int row, nrow = 0;
    int col, ncol = 0;
    int sumX, sumY = 0;
    int nIndex = 0;
    //printf("here: %d\n", 0);
    for (int index = 0; index < width*height; index ++) {
        
        row = index / height;
        col = index % width;
        sumX = 0;
        sumY = 0;
        //printf("here: %d\n", 1);
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                nrow = row + j;
                ncol = col + i;
                if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
                    nIndex = nrow * width + ncol;
                    //printf("here: %d\n", 2);
                    sumX += orig->data[nIndex] * Kx[i+1][j+1];
                    sumY += orig->data[nIndex] * Ky[i+1][j+1];
                    //printf("here: %d\n", 3);
                }
            }
        }
        //printf("here: %d\n", 4);
        gradient->width = orig->width;
        gradient->height = orig->height;
        angle->width = orig->width;
        angle->height = orig->height;
        //printf("here: %d\n", 5);
        gradient->data[index] = (char)sqrt(sumX * sumX + sumY * sumY);
        angle->data[index] = (char)(atan2(sumY, sumX) * 180) / 3.1415; // pi macro?? 
        //printf("here: %d\n", 6);

    }
    
}


int nonMaximumSuppression(Frame *gradient, Frame *angle, Frame *res) {
    // every 45 degrees?
    int width = gradient->width;
    int height = gradient->height;
    int maxGradient = 0;
    int valley1, valley2;

    for (int i = 0; i < width * height; i++) {
        if (gradient->data[i] > maxGradient) maxGradient = gradient->data[i];
    }
    int strong = .5 * maxGradient;
    int strongVal = .75 * maxGradient;
    int weak = .05 * maxGradient;
    int row, col = 0;
    int g, a;

    for (int i = 0; i < width * height; i++) {
        g = (int)gradient->data[i];
        a = (int)angle->data[i] % 180;

        row = i / width;
        col = i % height;

        int index1, index2;

        //a bunch of if conditions and you set valley 1 and valley 2 based on where they lie
        if ((a >= 157.5) || (a < 22.5)) {
            index1 = (row + 1) * width + col;
            index2 = (row - 1) * width + col;
        } else if ((a >= 22.5) && (a < 67.5)) {
            index1 = (row + 1) * width + (col - 1);
            index2 = (row - 1) * width + (col + 1);
        } else if ((a >= 67.5) && (a < 112.5)) {
            index1 = row * width + (col + 1);
            index2 = row * width + (col - 1);
        } else {
            index1 = (row + 1) * width + (col + 1);
            index2 = (row - 1) * width + (col - 1);
        }

        valley1 = (gradient->data[index1]);
        valley2 = (gradient->data[index2]);
        if ((valley1 < g) && (valley2 < g)) {
            if (g >= strong) res->data[i] = strongVal;
            else if (g >= weak) res->data[i] = weak;
            else res->data[i] = 0;
        } else {
            res->data[i] = 0;
        }
        
    }
    return maxGradient;
}


// void doubleThreshold(Frame &orig, Frame &res, int lo, int hi);

void hysteresis(Frame *orig, Frame *res, int maxGradient) {

    int width = orig->width;
    int height = orig->height;

    int strong =  .1 * maxGradient;
    int strongVal = .75 * maxGradient;
    int weak = .05 * maxGradient;
    int row, col, nrow, ncol;
    int nIndex;
    bool strongExists = false;

    for (int i = 0; i < width * height; i++) {

        row = i / width;
        col = i % height;
        if (orig->data[i] > weak) {
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    nrow = row + j;
                    ncol = col + i;
                    if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
                        nIndex = nrow * width + ncol;
                        if (orig->data[nIndex] >= strongVal) {
                            strongExists = true;
                        }
                    }
                }
            }
            if (strongExists) res->data[i] = strongVal;
            else res->data[i] = 0;
        } else if (orig->data[i] > strong) res->data[i] = strongVal;
        strongExists = false;
    }
}