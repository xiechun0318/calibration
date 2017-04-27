#pragma once
#include "opencv.hpp"
#include "mStaticFunc.h"

class mOptimizer
{
public:

	//result
	struct Result {
		double k1;
		double k2;
		double energy;
		double cx;
		double cy;
	};
public:
	mOptimizer();
	~mOptimizer();

public:
	double energy(cv::Mat linePoints, const double & distCenterX = NULL, const double & distCenterY = NULL);
	void extractLines(const cv::Mat& corners, std::vector<cv::Mat> & lines, cv::Size& size, uint8_t flag = ext_line_row| ext_line_col);
	int minEnergyLine(std::vector<cv::Mat> & lines);
	double energyAll(const cv::Mat& corners, cv::Size& patternSize, const double& distCenterX = NULL, const double &distCenterY = NULL);
	Result SA(cv::Mat& proj_pattern, cv::Size& patternSize);//Simulated Annealing
	double SA_F(cv::Mat& proj_pattern, cv::Size& patternSize, double k1_w, double k2_w);
	double SA_F(cv::Mat& proj_pattern, cv::Size& patternSize, std::vector<double> k1k2_w);
	void findDistCenter(cv::Mat& proj_pattern, cv::Size& patternSize, cv::Point2d& center);
	void init(double k1_p, double k2_p, cv::Size size);
	cv::Mat src_corners;

public:
	//flags
	static const uint8_t ext_line_row = 0x01;
	static const uint8_t ext_line_col = 0x02;

private:

	//distortion map for projection
	cv::Mat map_x_p, map_y_p;
	cv::Point2d distCenter_gt, distCenter_est, capturedDistCenter_est;
	cv::Mat hg = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
	                                    0.0, 1.0, 0.0,
										0.0,0.0,1.0);

};

