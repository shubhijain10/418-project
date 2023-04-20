/*
take an image
step 1: gaussian kernel
step 2: sobel filters using gradient calculation
step 3: non-maximum suppression
step 4: double threshold
step 5: hysteresis
*/
#include <iostream>
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
    // printf("height = %d\n", orig.height);
    // fflush(stdout);
    // printf("width = %d\n", orig.width);
    // fflush(stdout);
    // launch a kernel here
    for (int index = 0; index < width*height; index ++) {
        
        y = index / width;
        x = index % height;
        weightedSum = 0;
        sum = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                nx = x + i;
                ny = y + j;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    pixel_value = orig->data[ny * width + nx];
                    
                    weightedSum += pixel_value * kernel3[i+1][j+1];
                    //printf("pixel value: %d\n", pixel_value);
                    sum += kernel3[i+1][j+1];

                }
                
            }
        }
        // printf("index: %d\n", index);
        printf("sum: %d\n", sum);
        printf("weighted sum: %d\n", weightedSum);
        // fflush(stdout);
        unsigned char result = weightedSum/sum;
        printf("res %d\n", result);
        res->data[index] = (unsigned char)result;
        // printf("index: %d\n", index);
    }
    cout << "done going through the thing\n";

    // res.height = height;
    // res.width = width;

    // return res;
}


// void sobelFilters(Frame *orig, Frame *gradient, Frame *angle) {

//     int width = orig.width;
//     int height = orig.height;
//     int row, nrow = 0;
//     int col, ncol = 0;
//     int sumX, sumY = 0;

//     for (int index = 0; index < width*height; index ++) {
        
//         row = index / height;
//         col = index % width;
//         sumX, sumY = 0;
        
//         for (int i = -1; i <= 1; i++) {
//             for (int j = -1; j <= 1; j++) {
//                 nRow = row + j;
//                 nCol = col + i;
//                 if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
//                     nIndex = nRow * width + nCol;
//                     sumX += orig.data[index] * Kx[i+1][j+1];
//                     sumY += orig.data[index] * Kx[i+1][j+1];
//                 }
//             }
//         }

//         gradient.width = orig.width;
//         gradient.height = orig.height;
//         angle.width = orig.width;
//         angle.height = orig.height;

//         gradient.data[index] = (char)sqrt(sumX * sumX + sumY * sumY);
//         angle.data[index] = (char)(atan2(sumY, sumX) * 180) / 3.1415; // pi macro?? 
        

//     }
// }


// void nonMaximumSuppression(Frame &gradient, Frame &angle, Frame &res) {
    // every 45 degrees?
    // int width = gradient.width;
    // int height = gradient.height;
    // int gradient, angle, maxGradient = 0;
// 
    // for (int i = 0; i < width * height; i++) {
        // if (gradient.data[i] > maxGradient) maxGradient = gradient.data[i];
    // }
    // int strong = .1 * maxGradient;
    // int weak = .05 * maxGradient;
    // int row, col = 0;
// 
    // for (int i = 0; i < width * height; i++) {
        // g = (int)gradient.data[i];
        // angle = (int)angle.data[i] % 180;
// 
        // row = i / width;
        // col = i % height;
// 
        // int index1, index2;
// 
        // a bunch of if conditions and you set valley 1 and valley 2 based on where they lie
        // if ((angle >= 157.5) || (angle < 22.5)) {
            // index1 = (row + 1) * width + col;
            // index2 = (row - 1) * width + col;
        // } else if ((angle >= 22.5) && (angle < 67.5)) {
            // index1 = (row + 1) * width + (col - 1);
            // index2 = (row - 1) * width + (col + 1);
        // } else if ((angle >= 67.5) && (angle < 112.5)) {
            // index1 = row * width + (col + 1);
            // index2 = row * width + (col - 1);
        // } else {
            // index1 = (row + 1) * width + (col + 1);
            // index2 = (row - 1) * width + (col - 1);
        // }
// 
        // valley1 = (int)gradient.data[index1];
        // valley2 = (int)gradient.data[index2];
        // if ((valley1 < gradient) && (valley2 < gradient)) {
            // if (g >= strong) res.data[i] = strong;
            // else if (g >= weak) res.data[i] = weak;
            // else res.data[i] = 0;
        // } else {
            // res.data[i] = 0;
        // }
        // 
    // }
// }


// void doubleThreshold(Frame &orig, Frame &res, int lo, int hi);

// void hysteresis(Frame &orig, Frame &res, int lo, int hi);