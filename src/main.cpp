#include <iostream>
#include "include/global.hpp"

using namespace std;
// using namespace cv;

int height, width;
unsigned char type[10];
int intensity;

int main(int argc, char* argv[]) {
    // the make file will pass in argument for file name
    // take file and get the pixel data struct for it
    // pass it into the overall canny filter
    // save the output as an image
    // string image_name;
    // for (int i = 1; i < argc; i++) {
    //   if (strcmp(argv[i], "-f") == 0)
    //     image_name = argv[i + 1];
    // }
    
    // string image_path = image_name;
    
    string image_path = "./square.pgm";
    ifstream ifimage(image_path, ios::binary);
    ofstream ofimage("./output_images/canny_cuda_img.pgm", ios::binary);
    if (!ifimage.is_open()) {
        cout << "image file is empty";
        return -1;
    }

    // get the image information from the file
    ifimage >> ::type >> ::width >> ::height >> ::intensity;
    ofimage << type << endl << width << " " << height << endl << intensity << endl;

    // Frame *image = new Frame();
    unsigned char* image = new unsigned char[width*height];
    unsigned char* finished = new unsigned char[width*height];
    for (int i = 0; i < (width*height); i++) {
        image[i] = (unsigned char)ifimage.get();
    }

    cudaCanny(image, width, height, finished);

    //apply gaussian blur
    // Frame *gaussIm = new Frame();
    // Frame *gradient = new Frame();
    // Frame *angle = new Frame();
    // Frame *suppressed = new Frame();
    // Frame *finished = new Frame();

    // gaussIm->data = new unsigned char[width*height];
    // gradient->data = new unsigned char[width*height];
    // angle->data = new unsigned char[width*height];
    // suppressed->data = new unsigned char[width*height];
    // finished->data = new unsigned char[width*height];

    // gaussIm->height = height;
    // gradient->height = height;
    // angle->height = height;
    // suppressed->height = height;
    // finished->height = height;

    // gaussIm->width = width;
    // gradient->width = width;
    // angle->width = width;
    // suppressed->width = width;
    // finished->width = width;

    // int maxGradient;

    // gaussianBlur(image, gaussIm);
    // sobelFilters(gaussIm, gradient, angle);
    // maxGradient = nonMaximumSuppression(gradient, angle, suppressed);
    // hysteresis(suppressed, finished, maxGradient);

    for (int r = 0; r < (width*height); r++) {
        ofimage << finished[r];
    }

    return 0;

    //now filter the image

}