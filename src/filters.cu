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
#include "include/global.hpp"

using namespace std;
#define kernel_size 3

// __constant__ Frame ImageA, ImageB;

__constant__ int kernel3[3][3] = {1, 2, 1,
                2, 4, 2,
                1, 2, 1};

__constant__ int kernel5[5][5] = {1, 4, 7, 4, 1,
                4, 16, 26, 16, 4,
                7, 26, 41, 26, 7,
                4, 16, 26, 16, 4,
                1, 4, 7, 4, 1};

__constant__ int Kx[3][3] = {-1, 0, 1,
            -2, 0, 2,
            -1, 0, 1};

__constant__ int Ky[3][3] = {1, 2, 1,
            0, 0, 0,
            -1, -2, -1};



__global__ void gaussianBlur(unsigned char* orig, unsigned char* res, int width, int height) {
    // cout << "in blurring mode\n";
    // int width = orig->width;
    // int height = orig->height;
    
    unsigned char pixel_value = 0;
    // int res_value = 0;
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
                    pixel_value = orig[ny * width + nx];
                    
                    weightedSum += pixel_value * kernel5[i+2][j+2];
                    //printf("pixel value: %d\n", pixel_value);
                    sum += kernel5[i+2][j+2];
                }
            }
        }
        unsigned char result = weightedSum/sum;
        res[index] = (unsigned char)result;
    }
    // cout << "done going through the thing\n";
}


__global__ void sobelFilters(unsigned char *orig, unsigned char *gradient, unsigned char *angle, int width, int height) {

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
                    sumX += orig[nIndex] * Kx[i+1][j+1];
                    sumY += orig[nIndex] * Ky[i+1][j+1];
                    //printf("here: %d\n", 3);
                }
            }
        }
        //printf("here: %d\n", 4);
        
        //printf("here: %d\n", 5);
        gradient[index] = (char)sqrt((float)(sumX * sumX + sumY * sumY));
        angle[index] = (char)(atan2((float)sumY, (float)sumX) * 180) / 3.1415; // pi macro?? 
        //printf("here: %d\n", 6);

    }
    
}


__global__ void nonMaximumSuppression(unsigned char *gradient, unsigned char *angle, unsigned char *res, int width, int height) {
    // every 45 degrees?
   
    int maxGradient = 0;
    int valley1, valley2;

    for (int i = 0; i < width * height; i++) {
        if (gradient[i] > maxGradient) maxGradient = gradient[i];
    }

    unsigned char strong = .3 * maxGradient;
    unsigned char strongVal = .75 * maxGradient;
    unsigned char weak = .1 * maxGradient;
    int row, col = 0;
    int a;
    unsigned char g;

    for (int i = 0; i < width * height; i++) {
        g = gradient[i];
        a = (int)angle[i] % 180;
        

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

        valley1 = gradient[index1];
        valley2 = gradient[index2];
        if ((valley1 < g) && (valley2 < g)) {
            if (g >= strong) {
                res[i] = strongVal;
                printf("writing strong\n");
            } else if (g >= weak) { 
                res[i] = weak;
                printf("writing weak\n");
            } else { 
                res[i] = 0;
                printf("writing zero\n");
            }
        } else {
            res[i] = 0;
        }
        
    }
}


// void doubleThreshold(Frame &orig, Frame &res, int lo, int hi);

__global__ void hysteresis(unsigned char *orig, unsigned char *res, unsigned char *gradient, int width, int height) {


    int maxGradient = 0;

    for (int i = 0; i < width * height; i++) {
        if (gradient[i] > maxGradient) maxGradient = gradient[i];
    }

    int strong =  (int)(.3 * maxGradient);
    int strongVal = (int)(.75 * maxGradient);
    int weak = (int)(.1 * maxGradient);
    int row, col, nrow, ncol;
    int nIndex;
    bool strongExists = false;

    for (int index = 0; index < width * height; index++) {

        row = index / width;
        col = index % height;
        if (orig[index] > weak) {
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    nrow = row + j;
                    ncol = col + i;
                    if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
                        nIndex = nrow * width + ncol;
                        if (orig[nIndex] >= strongVal) {
                            strongExists = true;
                        }
                    }
                }
            }
            if (strongExists) res[index] = strongVal;
            else res[index] = 0;
        } else if (orig[index] > strong) res[index] = strongVal;
        strongExists = false;
    }
}

void cudaCanny(unsigned char* inImage, int width, int height, unsigned char* outImage) {
    unsigned char* device_image_orig;
    unsigned char* device_image_gaussian;
    unsigned char* device_image_gradient;
    unsigned char* device_image_angle;
    unsigned char* device_image_suppressed;
    unsigned char* device_image_hysteresis;


    cudaMalloc((void **)&(device_image_orig), sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&(device_image_gaussian), sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&(device_image_gradient), sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&(device_image_angle), sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&(device_image_suppressed), sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&(device_image_hysteresis), sizeof(unsigned char) * width * height);

    cudaMemcpy(device_image_orig, inImage, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    gaussianBlur<<<1, 1>>>((unsigned char*)device_image_orig, 
                            (unsigned char*)device_image_gaussian, 
                            width, height);

    sobelFilters<<<1, 1>>>((unsigned char*)device_image_gaussian, 
                            (unsigned char*)device_image_gradient, 
                            (unsigned char*)device_image_angle, 
                            width, height);
    
    nonMaximumSuppression<<<1, 1>>>(device_image_gradient, 
                                    device_image_angle,  
                                    device_image_suppressed,
                                    width, height);

    hysteresis<<<1, 1>>>((unsigned char*)device_image_suppressed,
                            (unsigned char*)device_image_hysteresis,  
                            (unsigned char*)device_image_gradient,
                            width, height);

    cudaMemcpy(outImage, (void *)device_image_hysteresis, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
    
    cudaFree((void *)device_image_orig);
    cudaFree((void *)device_image_gaussian);
    cudaFree((void *)device_image_gradient);
    cudaFree((void *)device_image_angle);
    cudaFree((void *)device_image_suppressed);
    cudaFree(device_image_hysteresis);
}