#include <iostream>
#include "include/global.hpp"
#include "include/timing.hpp"
#include <omp.h>

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
    omp_set_dynamic(0);     
    omp_set_num_threads(4);

    string image_path = "./lena.pgm";
    ifstream ifimage(image_path, ios::binary);
    ofstream ofimage("./output_images/canny_openmp_img.pgm", ios::binary);
    if (!ifimage.is_open()) {
        cout << "image file is empty";
        return -1;
    }

    // get the image information from the file
    ifimage >> ::type >> ::width >> ::height >> ::intensity;
    ofimage << type << endl << width << " " << height << endl << intensity << endl;

    Frame *image = new Frame();
    image->data = new unsigned char[width*height];
    image->height = height;
    image->width = width;

    for (int i = 0; i < (width*height); i++) {
        image->data[i] = (unsigned char)ifimage.get();
    }

    //apply gaussian blur
    Frame *gaussIm = new Frame();
    Frame *gradient = new Frame();
    Frame *angle = new Frame();
    Frame *suppressed = new Frame();
    Frame *finished = new Frame();

    gaussIm->data = new unsigned char[width*height];
    gradient->data = new unsigned char[width*height];
    angle->data = new unsigned char[width*height];
    suppressed->data = new unsigned char[width*height];
    finished->data = new unsigned char[width*height];

    gaussIm->height = height;
    gradient->height = height;
    angle->height = height;
    suppressed->height = height;
    finished->height = height;

    gaussIm->width = width;
    gradient->width = width;
    angle->width = width;
    suppressed->width = width;
    finished->width = width;

    int maxGradient;

    Timer totalSimulationTimer;
    gaussianBlur(image, gaussIm);
    sobelFilters(gaussIm, gradient, angle);
    maxGradient = nonMaximumSuppression(gradient, angle, suppressed);
    hysteresis(suppressed, finished, maxGradient);

    double totalSimulationTime = totalSimulationTimer.elapsed();
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    int curr_num = omp_get_num_threads();
    printf("threads:%d\n", curr_num);

    for (int r = 0; r < (width*height); r++) {
        ofimage << finished->data[r];
    }

    return 0;

    //now filter the image

}