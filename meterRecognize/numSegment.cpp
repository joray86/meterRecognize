#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


vector<Mat> numSegment(Mat img)
{
	Mat gray, th, hsv;
	medianBlur(img, img, 3);


	cvtColor(img, gray, CV_BGR2GRAY);
	
	//�������ᵽ������Ӧ��ֵ�ָ�Ч���Ϻ�,�ڸ���ͨ����𲻴��ûҶ�ͼ���ָ�Ϳ���
	adaptiveThreshold(gray, th, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 11, 15);

	//�ָ��ROI����
	vector <vector<Point>> contours;
	findContours(th, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> contour_tmp = contours[0];
	for (int i = 1; i < contours.size(); i++)
	{
		if (contourArea(contour_tmp) < contourArea(contours[i]))
			contour_tmp = contours[i];
	}

	//  �ָ�ROI����
	RotatedRect box = minAreaRect(contour_tmp);
	Point2f vertex[4];
	box.points(vertex);

	// �ָ��ROI������Ϊbox��һ����ת�ľ��Σ������ȶ�ͼ����һ�±任
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
	adaptiveThreshold(ROI_cropped, RoiTh, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 11, 15);

	//��2. �½�һ����Roiͬ����С�Ŀ�ͼ��RoiThOut������Roith�����������ͨ����ģ���RoiThOut�ϻ�����
	//findcontours����ȡ������������ôȡ��һ��Բ���ܻᱻʶ����������������������ڵ������ǣ�
	//�е�ͼƬ����������������ķֽ���ճ���������������ɾ�����п��ܻ���еķֽ���Ҳɾ�������һ����ܿ��ƣ���ȡ���������ͼƬ��
	//������һ�θ�ʴ����ôԭ�е�����ͼ����Ϣ�ͻ�ɥʧһ�㡣
	Mat RoiThOut = Mat::zeros(RoiTh.rows, RoiTh.cols, CV_8UC1);
	Mat RoiThClone = RoiTh.clone();
	vector <vector<Point>> contoursRoi;
	findContours(RoiTh, contoursRoi, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	vector<Point> contour_tmp_Roi = contoursRoi[0];
	int iterOut = 0;
	int iterIn = 0;
	for (int i = 0; i < contoursRoi.size(); i++)
	{
		double area = contourArea(contoursRoi[i]);
		if (contourArea(contour_tmp_Roi) <  contourArea(contoursRoi[i])) {
			contour_tmp_Roi = contoursRoi[i];
			iterOut = i;
		}
		else if (contourArea(contoursRoi[iterIn]) < area)
		{
			iterIn = i;
		}
	}

	for (int i = 0; i < contoursRoi.size(); i++)
	{
		Rect brect = boundingRect(contoursRoi[i]);
		if (i != iterOut & i != iterIn & brect.width>RoiThOut.cols / 20 & brect.height < RoiThOut.rows * 2 / 3) {
			drawContours(RoiThOut, contoursRoi, i, 255, -1);
		}
	}

	//�ָ���ַ�
	// ��1. ȥ������ô�ֱͶӰ���ָ�
	int* v = new int[RoiThOut.cols];
	int* h = new int[RoiThOut.rows];
	//cout << RoiThOut.cols << "...." << RoiThOut.rows;

	//��x��ͶӰ
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
	}
	// ��x��ͶӰ������߽�����vx����Ϊx��ķֽ���
	vector<int> vx1, vx2;
	for (int i = 0; i < RoiThOut.cols - 1; i++) {
		if (v[i] == 0 && v[i + 1] != 0)
			vx1.push_back(i);
		else if (v[i] != 0 && v[i + 1] == 0)
			vx2.push_back(i);
	}


	//��y��ͶӰ
	Mat My = Mat::zeros(RoiThOut.rows, RoiThOut.cols, CV_8UC1);

	for (int y = 0; y < RoiThOut.rows; y++) {
		data = RoiThOut.ptr<uchar>(y);
		h[y] = 0;
		for (int x = 0; x < RoiThOut.cols; x++) {
			if ((int)data[x] == 255)
				h[y] += 1;
		}

	}
	//��y��ͶӰ������߽�����vy����Ϊy��ķֽ���
	vector<int> vy1, vy2;
	for (int j = 0; j < RoiThOut.rows - 1; j++) {
		if (h[j] == 0 && h[j + 1] != 0)
			vy1.push_back(j);
		else if (h[j] != 0 && h[j + 1] == 0)
			vy2.push_back(j);
	}

	vector<Mat> numMat;
	Mat MatTemp;
	for (int x = 0; x < 5; x++) {
		MatTemp = RoiThClone(Rect(vx1[x], vy1[0], vx2[x] - vx1[x], vy2[0] - vy1[0]));
		numMat.push_back(MatTemp);
	}
	
	return numMat;
}

int main() {
	Mat img;
	img = imread("C:/Users/JoRay/Desktop/�Ӿ�ʶ��ˮ��/test/meter4.jpg");
	vector<Mat> numMat = numSegment(img);

	for (int x = 0; x < numMat.size(); x++) {

		ostringstream name;
		name << "C:/Users/JoRay/Desktop/�Ӿ�ʶ��ˮ��/test/numMat/" << x << ".png";
		imwrite(name.str(), numMat[x]);
	}
}