#pragma once
#include "opencv.hpp"

static cv::Mat Hpc, Hcf;
static int patterntype = 0;

static cv::Mat Hpc0, Hcf0, Hpc1, Hcf1, Hpc2, Hcf2, Hpc3, Hcf3;

static cv::Mat_<double> temp_scale = (cv::Mat_<double>(3, 3) << 1.0, 0.0, -50.0,
	0.0, 1.0, -100.0,
	0.0, 0.0, 0.7);
static cv::Mat_<double> temp_rot = (cv::Mat_<double>(3, 3) << 0.85, -0.85, 0.0,
	0.85, 0.85, 0.0,
	0.0, 0.0, 1);
static cv::Mat_<double> temp_shift = (cv::Mat_<double>(3, 3) << 1.0, 0.0, -512.0,
	0.0, 1.0, -384.0,
	0.0, 0.0, 1.0);
static cv::Mat_<double> temp_shiftback = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 512.0,
	0.0, 1.0, 384.0,
	0.0, 0.0, 1.0);

static cv::Mat  rot_mat = temp_shiftback*temp_rot*temp_shift;

static inline void readHomography() {
	cv::FileStorage fs("homography.xml",cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "homography.xml not found" << std::endl;
		return;
	}
	fs["H01"] >> Hpc;
	fs["H02"] >> Hcf;

	fs["H01"] >> Hpc0;
	fs["H02"] >> Hcf0;
	fs["H11"] >> Hpc1;
	fs["H12"] >> Hcf1;
	fs["H21"] >> Hpc2;
	fs["H22"] >> Hcf2;
	fs["H31"] >> Hpc3;
	fs["H32"] >> Hcf3;
}

static inline void createChessBoard(cv::Mat& out, const int& gridSize = 64) {

	out.create(768, 1024, CV_8UC1);
	//int gridSize = 64;
	if (patterntype == 0) {
		for (int i = 0; i < 768; i++) {
			for (int j = 0; j < 1024; j++) {
				out.at<char>(i, j) = 255;
			}
		}
		for (int i = 0; i < 768 - gridSize; i++) {
			for (int j = 0; j < 1024 - gridSize; j++) {
				if ((i % (gridSize * 2) < gridSize && j % (gridSize * 2) < gridSize) || (i % (gridSize * 2) >= gridSize && j % (gridSize * 2) >= gridSize)) {
					out.at<char>(i + gridSize / 2, j + gridSize / 2) = 0;
				}
				else {
					out.at<char>(i + gridSize / 2, j + gridSize / 2) = 255;
				}
			}
		}
	}
	else if (patterntype == 1) {
		for (int i = 0; i < 768; i++) {
			for (int j = 0; j < 1024; j++) {
				out.at<char>(i, j) = 255;
			}
		}
		for (int i = gridSize / 2 + gridSize; i <= 768 - gridSize / 2 - gridSize; i += gridSize) {
			for (int j = gridSize / 2 + gridSize; j <= 1024 - gridSize / 2 - gridSize; j += gridSize) {
				cv::circle(out, cv::Point2i(j, i), 10, cv::Scalar(0), -1);
			}
		}
	}
}

static inline bool extractCorners(cv::Mat& pattern, cv::Size& patternSize, cv::Mat& corners) {
	bool patternfound = false;
	if (patterntype == 0) {
		patternfound = findChessboardCorners(pattern, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
			| cv::CALIB_CB_FAST_CHECK);
		if (patternfound)
			cornerSubPix(pattern, corners, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	else if (patterntype == 1) {
		patternfound = findCirclesGrid(pattern, patternSize, corners);
	}
	return patternfound;
}

static inline void createMap(cv::Mat& map_x, cv::Mat& map_y, const double& k1, const double& k2,const double cx, const double cy, const double scale = 1)
{
	double r, r_2, r_4;
	for (int i = 0; i < map_x.rows; i++) {
		for (int j = 0; j < map_x.cols;j++) {
			r_2 = pow((j - cx), 2) + pow((i - cy), 2);
			r_4 = pow(r_2, 2);
			//r = sqrt(r_2);
			map_x.at<float>(i, j) = scale*(1 + k1*r_2 + k2*r_4)*(j - cx) + cx;
			map_y.at<float>(i, j) = scale*(1 + k1*r_2 + k2*r_4)*(i - cy) + cy;
		}
	}
}

static inline void createMap(cv::Mat& map_x, cv::Mat& map_y, const double& k1, const double& k2, const double scale = 1)
{
	double cx, cy;
	cx = map_x.cols / 2.f;
	cy = map_x.rows / 2.f;
	createMap(map_x, map_y, k1, k2, cx, cy, scale);
}


static inline void createInverseMap(cv::Mat& map_x, cv::Mat& map_y, const double& k1, const double& k2)
{
	double r, r_2, cx, cy;
	cx = map_x.cols / 2.f;
	cy = map_x.rows / 2.f;
	for (int i = 0; i < map_x.rows; i++) {
		for (int j = 0; j < map_x.cols;j++) {
			r_2 = pow((j - cx), 2) + pow((i - cy), 2);
			r = sqrt(r_2);
			map_x.at<float>(i, j) = (j - cx) / (1 + k1*pow(r, 2) + k2*pow(r, 4)) + cx;
			map_y.at<float>(i, j) = (i - cy) / (1 + k1*pow(r, 2) + k2*pow(r, 4)) + cy;
		}
	}
}

static inline cv::Mat translateImg(cv::Mat &img_in,cv::Mat &img_out, int offsetx, int offsety) {
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img_in, img_out, trans_mat, img_in.size(),1,0,cv::Scalar(255,255,255));
	return trans_mat;
}

static inline void capture_simulate(cv::Mat &pattern, cv::Mat &distored, cv::Mat &captured, cv::Mat &map_x_p, cv::Mat &map_y_p) {
	cv::remap(pattern, distored, map_x_p, map_y_p, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
	//cv::warpPerspective(distored, captured, Hpc*rot_mat, cv::Size(640, 480));
	cv::warpPerspective(distored, captured, Hpc, cv::Size(640, 480));

	//temp
	/*
	cv::Mat map_x, map_y;
	map_x.create(cv::Size(640, 480), CV_32FC1);
	map_y.create(cv::Size(640, 480), CV_32FC1);

	double k1 = 7e-7;
	double k2 = -3e-14;
	createMap(map_x, map_y, k1, k2);
	cv::Mat dist_cap;
	*/
	//remap(captured, captured, map_x, map_y, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(150));

}

static inline void toCapturedCoordinate(cv::Point2d &src, cv::Point2d &dst) {
	std::vector<cv::Point2d> buf_src(1);
	buf_src[0] = src;
	std::vector<cv::Point2d> buf_dst(1);
	cv::perspectiveTransform(buf_src, buf_dst, Hpc*rot_mat);
	dst = buf_dst.at(0);
}