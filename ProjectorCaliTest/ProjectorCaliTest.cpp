// ProjectorCaliTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "mStaticFunc.h"

using namespace std;
using namespace cv;

int main()
{
	mOptimizer opt;

	Mat src;
	createChessBoard(src,64);
	readHomography();


	// temp use
	cv::Mat linePoints = (cv::Mat_<double>(2, 3) << 0.0, 2.5, 5.0,
	                                    0.0, 1.0, 0.0);
	cv::Mat convr, mean;

	//convar
	cv::calcCovarMatrix(linePoints, convr, mean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
	//std::cout << convr << std::endl;
	double det = cv::determinant(convr);
	std::cout << "det: " << det << std::endl;
/*
	Mat pattern = imread("pattern.png");
	resize(pattern, pattern, Size(2400, 1800));
	Mat rt,lt,rb,lb,temp;
	Mat trans_mat;

	warpPerspective(pattern(Rect(0, 0, 1200, 900)), lt, Hpc3.inv()*Hcf3.inv(), Size(1024, 768));

	trans_mat = (cv::Mat_<double>(3, 3) << 1, 0, 1200,
											0, 1, 0,
											0, 0, 1);
	warpPerspective(pattern(Rect(1200, 0, 1200, 900)), rt, Hpc2.inv()*Hcf2.inv()*trans_mat, Size(1024, 768));

	trans_mat = (cv::Mat_<double>(3, 3) << 1, 0, 0,
											0, 1, 900,
											0, 0, 1);
	warpPerspective(pattern(Rect(0, 900, 1200, 900)), lb, Hpc1.inv()*Hcf1.inv()*trans_mat, Size(1024, 768));

	trans_mat = (cv::Mat_<double>(3, 3) << 1, 0, 1200,
											0, 1, 900,
											0, 0, 1);
	warpPerspective(pattern(Rect(1200, 900, 1200, 900)), rb, Hpc0.inv()*Hcf0.inv()*trans_mat, Size(1024, 768));


	imshow("lt", lt);
	imshow("rt", rt);
	imshow("lb", lb);
	imshow("rb", rb);

	waitKey();

	//end temp use
*/

	//namedWindow("projection");
	//imshow("origin", src);

	//Mat dummy;
	//opt.energy(dummy);

	waitKey();
	Mat map_x_p, map_y_p, map_x_w, map_y_w, distored, corrected, prewarped, map_x_inv, map_y_inv;
	//projection distortion
	double k1_p, k2_p;
	//prewarp parameter
	double k1_w, k2_w;

	distored.create(src.size(), src.type());
	map_x_p.create(src.size(), CV_32FC1);
	map_y_p.create(src.size(), CV_32FC1);
	map_x_w.create(src.size(), CV_32FC1);
	map_y_w.create(src.size(), CV_32FC1);
	map_x_inv.create(src.size(), CV_32FC1);
	map_y_inv.create(src.size(), CV_32FC1);

	
	/*
	k1_p = 1e-7;
	k2_p = -1e-14;
	k1_w = -0.888789e-7;
	k2_w = 1.23261e-14;
	*/

	k1_p = 0.7e-7;
	k2_p = -0.7e-14;
	k1_w = -6.05296e-8;
	k2_w =  8.74556e-15;
	//k1_w = -7.51401e-8;
	//k2_w = 8.2574e-15;
	double cx_gt, cy_gt, cx_est, cy_est;
	cx_gt = 200.5;
	cy_gt = 580.5;
	cx_est = 208.5;
	cy_est = 583.5;

	createMap(map_x_p, map_y_p, k1_p, k2_p,cx_gt,cy_gt);
	createMap(map_x_w, map_y_w, k1_w, k2_w,cx_est,cy_est);

	remap(src, prewarped, map_x_w, map_y_w, CV_INTER_LINEAR,BORDER_CONSTANT,Scalar(255));
	remap(prewarped, corrected, map_x_p, map_y_p, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
	remap(src, distored, map_x_p, map_y_p, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));


	Mat correct_cap, src_cap, temp;
	capture_simulate(src, temp, src_cap, map_x_p, map_y_p);
	capture_simulate(prewarped, temp, correct_cap, map_x_p, map_y_p);
	imshow("correct_cap", correct_cap);
	imshow("src_cap", src_cap);

	imshow("src", src);
	imshow("prewarped", prewarped);
	imshow("projection", corrected);
	imshow("distored", distored);
	waitKey();
	Mat dist_corners, src_corners, corr_corners;
	extractCorners(src, Size(14, 10), src_corners);
	resize(corrected, corrected, src.size() / 2);
	extractCorners(corrected, Size(14, 10), corr_corners);
	extractCorners(distored, Size(14, 10), dist_corners);

	corr_corners *= 2;
	double e = opt.energyAll(corr_corners, Size(14, 10));
	cout << e << endl;
	e = opt.energyAll(dist_corners, Size(14, 10));
	cout << e << endl;

	waitKey();

	Mat overlap(src.size(), CV_8UC3);
	overlap = Scalar(255, 255, 255);
	for (int i = 0;i < src_corners.rows; i++) {
		//circle(overlap,src_corners.at<Point2f>(i),10,Scalar(100,200,0), CV_FILLED);
		//circle(overlap, corr_corners.at<Point2d>(i) * 2, 5, Scalar(0, 0, 255));
		rectangle(overlap, corr_corners.at<Point2f>(i)  - Point2f(10, 10), corr_corners.at<Point2f>(i)  + Point2f(10, 10), Scalar(0, 0, 255),2);
		circle(overlap, dist_corners.at<Point2f>(i) , 10, Scalar(0, 0, 0),2);
	}

	imshow("overlap", overlap);

	waitKey();

	//dist center
	opt.src_corners = src_corners;
	opt.init(k1_p, k2_p, src.size());
	Point2d distCenter;
	opt.findDistCenter(src, Size(14, 10), distCenter);
	waitKey();


	//optimization
	opt.init(k1_p, k2_p, src.size());
	mOptimizer::Result res = opt.SA(src, Size(14, 10));

	//evaluation
	k1_w = res.k1;
	k2_w = res.k2;
	cx_est = res.cx;
	cy_est = res.cy;
	createMap(map_x_w, map_y_w, k1_w, k2_w, cx_est, cy_est);
	remap(src, prewarped, map_x_w, map_y_w, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
	remap(prewarped, corrected, map_x_p, map_y_p, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
	remap(src, distored, map_x_p, map_y_p, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
	imshow("prewarped", prewarped);
	imshow("projection", corrected);
	imshow("distored", distored);
	waitKey();
	Mat corners;
	resize(corrected, corrected, src.size() / 2);
	bool found = extractCorners(corrected, Size(14, 10), corners);
	corners *= 2;
	double energy = opt.energyAll(corners,Size(14,10),NULL,NULL);

	Mat corners2;
	found = extractCorners(distored, Size(14, 10), corners2);
	double energy2 = opt.energyAll(corners2, Size(14, 10), NULL, NULL);

	cout << "Undistorted energy = " << energy << endl;
	cout << "Distorted energy = " << energy2 << endl;


	/*
	//experiment loop 10*10 samples
	k1_w = 0;
	k2_w = 0;
	double k1_w_step = -1e-7;
	double k2_w_step = 1e-14;
	ofstream a_file("example.txt");

	for (int i = 0; i < 10;i++) {
		k1_w = i * k1_w_step;
		for (int j = 0;j < 10;j++) {
			k2_w = j * k2_w_step;

			createMap(map_x_w, map_y_w, k1_w, k2_w);
			remap(src, prewarped, map_x_w, map_y_w, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
			remap(prewarped, distored, map_x_p, map_y_p, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
			imshow("src -> prewarped", prewarped);
			imshow("prewarped -> distored", distored);

			Mat captured, captured_color;
			warpPerspective(distored, captured, Hpc, Size(640, 480));
			imshow("Camera", captured);

			//vector<Point2f> corners;
			Mat corners;
			bool patternfound = findChessboardCorners(captured, Size(14, 10), corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
				+ CALIB_CB_FAST_CHECK);
			if (patternfound)
				cornerSubPix(captured, corners, Size(11, 11), Size(-1, -1),
					TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			if (!patternfound) {
				a_file << k1_w << "\t" << k2_w << "\t" << "!!fail!!" << "\n";
			}
			else {
				double energy = opt.energyAll(corners, Size(14, 10));

				cvtColor(captured, captured_color, CV_GRAY2BGR);
				drawChessboardCorners(captured_color, Size(14, 10), Mat(corners), patternfound);
				imshow("Captured_color", captured_color);

				a_file << k1_w << "\t" << k2_w << "\t" << energy << "\n";
			}
			waitKey(1);
		}
	}
	a_file.close();
	*/
	waitKey();
    return 0;
}


