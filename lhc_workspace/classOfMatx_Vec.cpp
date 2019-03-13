#include <opencv2/core.hpp>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <list>

using namespace cv;
using namespace std;


int main(int argc, char const *argv[])
{
	Mat M(4,4,CV_8UC1,Scalar(2,2,3));
	
	double sums = 0;
	int cols = M.cols, rows = M.rows;
	if(M.isContinuous())
	{
		cols *= rows;
		rows = 1;
		cout<<"this is continous"<<endl;
	}

	cout<<"sum is : "<<M<<endl;
	// for(int i=0; i< rows; i++)
	// {
	// 	uchar* Mi = M.ptr<uchar>(i);
	// 	for(int j=0; j < cols; j++)
	// 		{ sum += Mi[j];
	// 		cout<< (int)M.at<uchar>(i,j) <<endl;
	// 	}
			
	// }
	M.pop_back(4);
	cout << M <<endl;


	std::list<int> mylist;
	int myint;

	std::cout << "Please enter some integers (enter 0 to end):\n";

	do {
	std::cin >> myint;
	mylist.push_back (myint);
	} while ( !(myint == 0));

	// std::cout<< "list is : " << mylist << std::endl;
	std::cout << "mylist stores " << mylist.size() << " numbers.\n";

	return 0;
}