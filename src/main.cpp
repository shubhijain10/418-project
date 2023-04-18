#include <iostream>

using namespace std;
using namespace cv;

int height, width;
char type[10];
int intensity;

int main(int argc, int* argv) {
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
    
    string image_path = "sample_image.jpg";
    ifstream ifimage(image_path, ios::binary);
    ofstream ofimage("./output_images/canny_img.pgm", ios::binary);
    if (!ifimage.is_open()) {
        cout << "image file is empty";
        return -1;
    }

    // get the image information from the file
    ifimage >> ::type >> ::width >> ::height >> ::intensity;
    ofimage << type << endl << width << " " << height << endl << intensity << endl;

    char* image = new char[width*height];

    for (int i = 0; i < (width*height); i++) {
        image[i] = (char)ifimage.get();
    }

    //apply gaussian blur
    char* gaussIm = new double[width*height];
    gaussianBlur(image, gaussIm);



    for (int r = 0; r < (width*height); r++) {
        ofimage << final[r];
    }

    return 0;

    //now filter the image

}