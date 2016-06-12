#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
	Mat img, gray, th, hsv;
	//读取图片，meter可以写成绝对路径或者放在工程目录下
	img = imread("C:/Users/JoRay/Desktop/视觉识别水表/test/meter3.jpg");
	medianBlur(img, img, 3);


	cvtColor(img, gray, CV_BGR2GRAY);
	//cvtColor(img, hsv, CV_BGR2HSV);


	//vector<Mat> splitBGR(img.channels());
	////分割通道，存储到splitBGR中
	//split(img, splitBGR);

	//vector<Mat> splitHSV(img.channels());
	////分割通道，存储到splitBGR中
	//split(img, splitHSV);

//	equalizeHist(splitBGR[2], splitBGR[2]);

	//OTSU阈值分割
//	threshold(splitBGR[2], th, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	//论文中提到用自适应阈值分割效果较好,在各个通道差别不大，用灰度图做分割就可以
	adaptiveThreshold(gray, th, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 11, 15);

	//分割出ROI区域
	vector <vector<Point>> contours;
	findContours(th, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);	
	vector<Point> contour_tmp = contours[0];
	for (int i = 1; i < contours.size(); i++)
	{
		if (contourArea(contour_tmp) < contourArea(contours[i]))
			contour_tmp = contours[i];
	}
//	drawContours(img, contours, -1, Scalar(0, 0, 255));
//  分割ROI区域
	RotatedRect box = minAreaRect(contour_tmp);
	Point2f vertex[4];
	box.points(vertex);
	//for (int i = 0; i < 4; i++)
	//	line(img, vertex[i], vertex[(i + 1) % 4], Scalar(0, 0, 255), 2, LINE_AA);
//	Mat Roi = img(Rect(vertex[0],vertex[2])); //要按照rotaterect分割roi
// 分割出ROI区域，因为box是一个旋转的矩形，所以先对图像做一下变换
	Mat RoiTh, ROIrotated, ROI_cropped;
	float angle = box.angle;
	Size rect_size = box.size;
	if (box.angle < -45.) {
		angle += 90.0;
		swap(rect_size.width, rect_size.height);
	}
	Mat M = getRotationMatrix2D(box.center, angle, 1.0);
	warpAffine(img, ROIrotated, M, img.size(), INTER_CUBIC);
	getRectSubPix(ROIrotated, rect_size, box.center, ROI_cropped);	
	cvtColor(ROI_cropped, ROI_cropped, CV_BGR2GRAY);
	//	threshold(Roi, RoiTh, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	adaptiveThreshold(ROI_cropped, RoiTh, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 11, 15);

	// 去掉仪表外边框
	// 法1：RoiThOut是RoiTh的副本，遍历得到最大连通区域，对该连通区域的外边界，drawContours画一个粗为10，颜色为黑的线把外边框覆盖
	//该方法取决于drawContours中线条的粗细thickness
	//thickness的宽度是圆环宽度的2倍，圆环的面积是外圈面积减内圈面积，即面积最大的连通区域面积减次大的连通区域面积，长短大约为图片的周长
	//据此大体估算thickness
	//Mat element = getStructuringElement(MORPH_RECT, Size(2,2));
	//dilate(RoiTh, RoiTh,element); //膨胀效果不好
//	Mat RoiThOut = RoiTh.clone();
//	vector <vector<Point>> contoursRoi;
//	findContours(RoiTh, contoursRoi, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//	vector<Point> contour_tmp_Roi = contoursRoi[0];
//	int iterOut = 0;
//	int iterIn = 0;
//	//int iter = 0;
//	int a = 10;
//	for (int i = 0; i < contoursRoi.size(); i++)
//	{
//		double area = contourArea(contoursRoi[i]);
//		cout << i << " : " << area << endl;
//		if (contourArea(contour_tmp_Roi) < area){
//			contour_tmp_Roi = contoursRoi[i];
//			//iter = i;
//			iterIn = iterOut;
//			iterOut	= i;
//		}
//		else if(contourArea(contoursRoi[iterIn]) < area)
//		{
//			iterIn = i;
//		}
//		Rect brect = boundingRect(contoursRoi[i]);
////		rectangle(RoiThOut, brect, Scalar(255, 0, 0),1);
//		if(brect.width<a)
//			drawContours(RoiThOut, contoursRoi, i, 0, -1);
//	}
//	//cout << "iter: " << iter << ",  " << "area: " << contourArea(contoursRoi[iter]) << endl;
//	cout << "interIn: " << iterIn << ",  " << "area: " << contourArea(contoursRoi[iterIn]) << endl;
//	cout << "interOut: " << iterOut << ",  " << "area: " << contourArea(contoursRoi[iterOut]) << endl;
//	int thickness = round((contourArea(contoursRoi[iterOut]) - contourArea(contoursRoi[iterIn])) / (RoiTh.rows + RoiTh.cols)) + 2;
//	drawContours(RoiThOut, contoursRoi, iterOut, 0, thickness);


	//法2. 新建一个和Roi同样大小的空图像RoiThOut，遍历Roith，不是最大连通区域的，在RoiThOut上画出来
	//findcontours就是取轮廓，无论怎么取，一个圆环总会被识别称外轮廓和内轮廓，现在的问题是，
	//有的图片会出现内轮廓与竖的分界线粘连，如果把内轮廓删除，有可能会把有的分界线也删掉，而且还不能控制，这取决于拍摄的图片；
	//或者做一次腐蚀，那么原有的数字图像信息就会丧失一点。
	Mat RoiThOut = Mat::zeros(RoiTh.rows, RoiTh.cols, CV_8UC1);
	Mat RoiThClone = RoiTh.clone();
	vector <vector<Point>> contoursRoi;
	findContours(RoiTh, contoursRoi, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	vector<Point> contour_tmp_Roi = contoursRoi[0];
//	drawContours(RoiThOut, contoursRoi, -1, 255, -1);
//	cout << contoursRoi.size() << endl;
	int iterOut = 0;
	int iterIn = 0;
	for (int i = 0; i < contoursRoi.size(); i++)
	{
		double area = contourArea(contoursRoi[i]);
		cout << i << " : " << area << endl;
		if (contourArea(contour_tmp_Roi) <  contourArea(contoursRoi[i])){
			contour_tmp_Roi = contoursRoi[i];
			iterOut = i;
		}else if(contourArea(contoursRoi[iterIn]) < area)
		{
			iterIn = i;
		}
	}
	cout << "iter: " << iterOut << endl;
	//最大连通区域选择错误！！！！
//	drawContours(RoiThOut, contoursRoi, iter, 255, -1);
	for (int i = 0; i < contoursRoi.size(); i++)
	{
		Rect brect = boundingRect(contoursRoi[i]);
		if ((i != iterOut) & (i != iterIn) & (brect.width>RoiThOut.cols/20) & (brect.height < RoiThOut.rows*2/3)) {

		//	double area = contourArea(contoursRoi[i]);

			drawContours(RoiThOut, contoursRoi, i, 255, -1);
		}
	}
	
	//分割出字符
	// 法1. 去除外框，用垂直投影法分割
	int* v = new int[RoiThOut.cols];
	int* h = new int[RoiThOut.rows];
	//cout << RoiThOut.cols << "...." << RoiThOut.rows;
	
	//在x轴投影
	int val;
	const uchar* data;
	Mat Mx = Mat::zeros(RoiThOut.rows, RoiThOut.cols, CV_8UC1);
	for (int x = 0; x < RoiThOut.cols; x++) {
		v[x] = 0;
		for (int y = 0; y < RoiThOut.rows; y++) {
			data = RoiThOut.ptr<uchar>(y);
			if ((int)data[x] == 255)
				v[x] += 1;
		}
	//	cout <<v[x]<<", ";
	}
	// 将x轴投影的区域边界点放入vx，作为x轴的分界线
	vector<int> vx1, vx2;
	for (int i = 0; i < RoiThOut.cols - 1; i++) {
		if (v[i] == 0 && v[i + 1] != 0)
			vx1.push_back(i);
		else if (v[i] != 0 && v[i + 1] == 0)
			vx2.push_back(i);
	}

	for (int i = 0; i < vx1.size(); i++)
		cout << i << ": " << vx1[i]<<", " <<vx2[i]<< endl;
	
	// 生成x轴的投影图
	for (int x = 0; x < RoiThOut.cols; x++) {
		for (int y = v[x]; y >0 ; --y) {
			Mx.at<uchar>(RoiThOut.rows-y,x) = 255;
		}
	}
	
	//在y轴投影
	Mat My = Mat::zeros(RoiThOut.rows, RoiThOut.cols, CV_8UC1);
	
	for (int y = 0; y < RoiThOut.rows; y++) {
		data = RoiThOut.ptr<uchar>(y);
		h[y] = 0;
		for (int x = 0; x < RoiThOut.cols; x++) {
			if ((int)data[x] == 255)
				h[y] += 1;
		}
	//	cout << h[y] << ", ";
	}
	//将y轴投影的区域边界点放入vy，作为y轴的分界线
	vector<int> vy1, vy2;
	for (int j = 0; j < RoiThOut.rows - 1; j++) {
		if (h[j] == 0 && h[j + 1] != 0)
			vy1.push_back(j);
		else if (h[j] != 0 && h[j + 1] == 0)
			vy2.push_back(j);
	}

	for (int i = 0; i < vy1.size(); i++)
		cout << i << ": " << vy1[i] <<", "<<vy2[i]<< endl;

	//生产y轴的投影图
	for (int y = 0; y <  RoiThOut.rows; y++) {
		for (int x = 0; x < h[y]; x++) {
			My.at<uchar>(y, x) = 255;
		}
	}
	vector<Mat> numMat;
	Mat MatTemp;
	for (int x = 0; x < 5; x++) {

		MatTemp = RoiThClone(Rect(vx1[x],vy1[0],vx2[x]-vx1[x],vy2[0]-vy1[0]));
		numMat.push_back(MatTemp);
		ostringstream name;
		name <<"C:/Users/JoRay/Desktop/视觉识别水表/test/num/"<< x <<".png";
		imwrite(name.str(), MatTemp);
	}
		
	imwrite("C:/Users/JoRay/Desktop/视觉识别水表/test/ROI_mx.png", RoiThOut);
	imshow("测试程序", RoiThClone);
	imwrite("C:/Users/JoRay/Desktop/视觉识别水表/test/mx.png", Mx);
	imshow("Mx", Mx);
	imwrite("C:/Users/JoRay/Desktop/视觉识别水表/test/my.png", My);
	imshow("My", My);
	waitKey(500000);
}