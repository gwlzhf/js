#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "svm_image.h"

#include <array>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

using namespace std;
using namespace cv;

int low_H = 25,low_S = 43,low_V = 46;
int high_H = 85,high_S = 255,high_V = 255;

void Erosion( int erosion_elem, int erosion_size,Mat &src);
void removeZeroLine(Mat &image_hsv,Mat &final_image);
string type2str(int type);


int main(int argc, char const *argv[])
{
	Mat image, image_hsv;
	// std::vector<string> filenames;
	// std::string path="/home/lhc/ASABE/cnn_image/1";

	// read_directory(path, filenames);
	// std::vector<string>::iterator v = filenames.begin();

	
    std::string imagePath ("/home/lhc/ASABE/cnn_image/2/example418.jpg");
    // imagePath.append(*(v+1));

    cout<<imagePath<<endl;
	if(argc > 1)
		imagePath=argv[1];

	

	image = imread(imagePath, IMREAD_COLOR);
	cout<<"image size is"<<image.size()<<"Before process image type is: "<<type2str(image.type())<<endl;
	if(image.empty())
		return -1;
	//covert image's color space from BGR2HSV, select green block and remove zero line
	cvtColor(image,image_hsv,COLOR_BGR2HSV);
	inRange(image_hsv, Scalar(low_H,low_S,low_V), Scalar(high_H,high_S,high_V), image_hsv);
	Erosion(0,1,image_hsv);
	cout<<"image size is"<<image.size()<<"After process image type is: "<<type2str(image_hsv.type())<<endl;


	Mat final_image;	
	removeZeroLine(image_hsv, final_image);
	cout<<"After process image'size is: "<<final_image.size()<<endl;
	Mat outputImage(24,24,CV_8UC1);
	cv::resize(final_image, outputImage, outputImage.size());


	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image",image);
	imshow("image_hsv", image_hsv);
	imshow("final_image", final_image);
	waitKey(0);
	return 0;

}

void removeZeroLine(Mat &image_hsv,Mat &final_image)
{
	Mat image_temp;
	//calculate summary value of every colums
	for(int i=0; i < image_hsv.rows; ++i)
	{
		// cout<<sum(image_hsv.row(i))[0]<<endl;
		if (sum(image_hsv.row(i))[0] > 0)
		  final_image.push_back(image_hsv.row(i));
	}

	final_image = final_image.t();

	for(int i=0; i < final_image.rows; i++)
	{
		if (sum(final_image.row(i))[0] > 0)
		  	image_temp.push_back(final_image.row(i));
	}

	final_image = image_temp.t();
	
}
void Erosion( int erosion_elem, int erosion_size,Mat &src )
{
  int erosion_type = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
  erode( src, src, element,Point(-1,-1),2);
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}