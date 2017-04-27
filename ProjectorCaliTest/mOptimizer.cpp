#include "stdafx.h"
#include "mOptimizer.h"


mOptimizer::mOptimizer()
{
}


mOptimizer::~mOptimizer()
{
}

double mOptimizer::energy(cv::Mat linePoints, const double & distCenterX, const double & distCenterY)
{
	// linepoints
	// x1 x2 x3 ...
	// y1 y2 y3 ...
	//linePoints = (cv::Mat_<double>(2, 3) << 1.0, 3.0, 5.0,
		//                                    2.0, 4.0, 6.0);
	cv::Mat convr, mean;

	if (distCenterX == NULL || distCenterY == NULL) {
		//convar
		cv::calcCovarMatrix(linePoints, convr, mean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
		//std::cout << convr << std::endl;
		double det = cv::determinant(convr);
		//std::cout << "Line Energy: " << det << std::endl;
		return det;
	}
	else {
		//weighted convar
		cv::Mat weighted_samples = linePoints.clone();
		double w, tmp_x, tmp_y, w_sum = 0;
		std::vector<double> weight;
		for (int i = 0; i < weighted_samples.cols; i++) {
			tmp_x = weighted_samples.col(i).at<float>(0);
			tmp_y = weighted_samples.col(i).at<float>(1);
			w = cv::norm(cv::Point2d(distCenterX, distCenterY) - cv::Point2d(tmp_x, tmp_y));
			//w = 1.0/w;
			weighted_samples.col(i) *= w;
			weight.push_back(w);
			w_sum += w;
		}
		cv::Mat means;
		//cv::reduce(weighted_samples, means, 1, CV_REDUCE_SUM);
		//means /= w_sum;
		cv::reduce(linePoints, means, 1, CV_REDUCE_AVG);
		//std::cout << means << std::endl;

		double Sxx = 0, Sxy = 0, Syy = 0;
		for (int i = 0; i < weighted_samples.cols; i++) {
			tmp_x = linePoints.col(i).at<float>(0);
			tmp_y = linePoints.col(i).at<float>(1);
			Sxx += weight.at(i) * pow((tmp_x - means.at<float>(0)), 2);
			Sxy += weight.at(i) * (tmp_x - means.at<float>(0)) * (tmp_y - means.at<float>(1));
			Syy += weight.at(i) * pow((tmp_y - means.at<float>(1)), 2);
		}
//		cv::calcCovarMatrix(weighted_samples, convr, mean, CV_COVAR_COLS | CV_COVAR_NORMAL);
		convr = (cv::Mat_<double>(2, 2) << Sxx, Sxy, Sxy, Syy);
		convr /= w_sum;
		//std::cout << convr << std::endl;
		
		double det = cv::determinant(convr);
		//std::cout << "Weighted Line Energy: " << det << std::endl;

		return det;
	}

}

void mOptimizer::extractLines(const cv::Mat & corners, std::vector<cv::Mat>& lines, cv::Size & size, const uint8_t flag)
{
	int w = size.width;
	int h = size.height;

	lines.clear();

	//extract rows
	if (flag & ext_line_row) {
		//std::cout << "extract rows"<< std::endl;
		for (int i = 0; i < h; i++) {
			cv::Mat line(2, w, CV_32F);
			for (int j = 0; j < w; j++) {
				cv::Point2f p = corners.at<cv::Point2f>(i * w + j);
				//std::cout << p << " -> ";
				//
				std::vector<cv::Point2f> temp_s{ p };
				std::vector<cv::Point2f> temp_d(1);
				cv::perspectiveTransform(temp_s, temp_d, hg);
				p = temp_d.at(0);
				//std::cout << p << std::endl;
				//
				line.at<float>(0, j) = p.x;
				line.at<float>(1, j) = p.y;
			}
			//std::cout << "Row " << i+1 <<"/"<< h << ":" << std::endl;
			//std::cout << line << std::endl;
			lines.push_back(line);
		}
	}

	//extract cols
	if (flag & ext_line_col) {
		//std::cout << "extract cols" << std::endl;
		for (int i = 0; i < w; i++) {
			cv::Mat line(2, h, CV_32F);
			for (int j = 0; j < h; j++) {
				cv::Point2f p = corners.at<cv::Point2f>(i + j * w);
				//std::cout << p << " -> ";
				//
				std::vector<cv::Point2f> temp_s{ p };
				std::vector<cv::Point2f> temp_d(1);
				cv::perspectiveTransform(temp_s, temp_d, hg);
				p = temp_d.at(0);
				//std::cout << p << std::endl;
				//
				line.at<float>(0, j) = p.x;
				line.at<float>(1, j) = p.y;
			}
			//std::cout << "Col " << i+1 << "/" << w << ":" << std::endl;
			//std::cout << line << std::endl;
			lines.push_back(line);
		}
	}

}

int mOptimizer::minEnergyLine(std::vector<cv::Mat>& lines)
{
	int min_i = 0;
	double min_e = std::numeric_limits<double>::max();
	double current;
	for (int i = 0; i < lines.size();i++) {
		current = energy(lines[i]);
		if (current < min_e) {
			min_e = current;
			min_i = i;
		}
		std::cout << "energy " << i << " = " << current << std::endl;
	}
	
	return min_i;
}

double mOptimizer::energyAll(const cv::Mat & corners, cv::Size& patternSize, const double& distCenterX, const double &distCenterY)
{
	std::cout << "Extract Lines..." << std::endl;
	std::vector<cv::Mat> lines;
	extractLines(corners, lines, patternSize);
	std::cout << "Extract Lines... finished" << std::endl;

	double e = 0;
	for (int i = 0;i < lines.size(); i++) {
		//std::cout << "Calculate energy of Line " << i+1 << "/"<< lines.size() << std::endl;
		e += energy(lines.at(i), distCenterX, distCenterY);
	}
	e = e / lines.size();

	//std::cout << "Total No. of lines: " << lines.size() << std::endl;
	//std::cout << "Total energy: " << e << std::endl;

	return e;
}


mOptimizer::Result mOptimizer::SA(cv::Mat & proj_pattern, cv::Size & patternSize)
{
	/*Simulated Annealing*/
	std::cout << "Running Simulated Annealing..." << std::endl;
	//Numbuer of cycles
	int n = 30;
	//Number of trials per cycle
	int m = 30;
	//Number of accepted solutions
	double na = 0.0;
	//Probability of accepting worse solution at the start
	double p1 = 0.7;
	//Probability of accepting worse solution at the end
	double p50 = 0.001;
	//Initial temperature
	double t1 = -1.0 / std::log(p1);
	//Final temperature
	double t50 = -1.0 / std::log(p50);
	//Fractional reduction every cycle
	double frac = std::pow((t50 / t1), (1.0 / (n - 1.0)));

	// Start location
	std::vector<double> x_start = { -1e-7, 1e-14 };

	//Initialize x
	std::vector<std::vector<double>> x(n+1);
	x[0] = x_start;
	std::vector<double> xi = x_start;
	na += 1.0;

	//Current best results so far
	std::vector<double> xc = x[0];
	double fc = SA_F(proj_pattern, patternSize,xi);
	std::vector<double>fs(n+1);
	fs[0] = (fc);
	//Current temperature
	double t = t1;
	//DeltaE Average
	double DeltaE_avg = 0.0;

	//lower bound of k1
	double lb_k1 = -10e-7;//-1e-6
	//upper bound of k1
	double ub_k1 = 0;
	//lower bound of k1
	double lb_k2 = 0;
	 //upper bound of k1
	double ub_k2 = 10e-14;//1e-13
	
	//initialize random seed
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> rand11(-1,1);//Range of step size -1 ~ 1
	std::uniform_real_distribution<> rand01(0, 1);//probability 0 ~ 1

	for (int i = 0;i < n;i++) {
		std::cout << "Cycle: " << i << " with Temperature: " << t << std::endl;
		for (int j = 0;j < n;j++) {
			//Generate new trial points
			xi[0] = xc[0] + rand11(mt)*1e-7;
			xi[1] = xc[1] + rand11(mt)*1e-14;
			//test
			if (i > n / 2.0) {
				xi[0] /= 2.0;
				xi[1] /= 2.0;
			}
			//end test
			//Clip to upper and lower bounds
			xi[0] = MAX(MIN(xi[0], ub_k1), lb_k1);
			xi[1] = MAX(MIN(xi[1], ub_k2), lb_k2);

			std::cout << "Trying: k1 = " << xi[0] << " k2 = " << xi[1] << std::endl;

			double fnew = SA_F(proj_pattern, patternSize, xi);
			if (fnew < 0) {
				j--;
				std::cout << "SA_F < 0, retry" << std::endl;
				continue;
			}
			double DeltaE = abs(fnew - fc);

			std::cout << "fnew = " << fnew << std::endl;
			std::cout << "DeltaE = " << DeltaE << std::endl;
			//accept flag
			bool accept;
			if (fnew > fc) {
				//Initialize DeltaE_avg if a worse solution was found
				//on the first iteration
				if (i == 0 && j == 0)
					DeltaE_avg = DeltaE;
				// objective function is worse
				// generate probability of acceptance
				double p = exp(-DeltaE / (DeltaE_avg * t));
				// determine whether to accept worse point
				double r = rand01(mt);
				if (r < p) {
					// accept the worse solution
					std::cout << "!!!Luck Jump r = " << r << " p = " << p <<" DeltaE = " << DeltaE <<" DeltaE_avg = "<< DeltaE_avg << std::endl;
					accept = true;
				}
				else {
					// don't accept the worse solution
					accept = false;
				}
			}
			else {
				//objective function is lower, automatically accept
				accept = true;
			}

			if (accept) {
				xc[0] = xi[0];
				xc[1] = xi[1];
				fc = fnew;
				//increment number of accepted solutions
				na = na + 1.0;
				//update DeltaE_avg
				DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na;
			}
		}

		//Record the best x values at the end of every cycle
		x[i + 1]= {xc[0], xc[1]};

		fs[i + 1] = fc;

		//reduce DeltaE_avg
		DeltaE_avg *= 0.5;

		std::cout << "Round "<< i << " Tempory result: k1 = " << xc[0] <<" k2 = " << xc[1] <<" energy = " << fc << std::endl;
		//Lower the temperature for next cycle
		t = frac * t;
	}
	std::cout << "Best Solution: k1 = " <<xc[0] <<" k2 = "<< xc[1] << std::endl;
	std::cout << "Best Objective: energy = " << fc << std::endl;
	std::cout << "Dist Center est = " << distCenter_est << std::endl;

	Result res;
	res.k1 = xc[0];
	res.k2 = xc[1];
	res.energy = fc;
	res.cx = distCenter_est.x;
	res.cy = distCenter_est.y;
	return res;
}

double mOptimizer::SA_F(cv::Mat & proj_pattern, cv::Size & patternSize, double k1_w, double k2_w)
{
	cv::Mat prewarped, distored, captured, captured_color;
	cv::Mat map_x_w, map_y_w;
	map_x_w.create(proj_pattern.size(), CV_32FC1);
	map_y_w.create(proj_pattern.size(), CV_32FC1);
	createMap(map_x_w, map_y_w, k1_w, k2_w, distCenter_est.x, distCenter_est.y);
	cv::remap(proj_pattern, prewarped, map_x_w, map_y_w, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
	//cv::remap(prewarped, distored, map_x_p, map_y_p, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
	
	capture_simulate(prewarped, distored, captured, map_x_p, map_y_p);
	cv::imshow("src -> prewarped", prewarped);
	cv::imshow("prewarped -> distored", distored);

	//warpPerspective(distored, captured, Hpc,cv::Size(640, 480));
	cv::imshow("Camera", captured);

	cv::Mat corners;
	bool found = extractCorners(captured, patternSize, corners);

	cvtColor(captured, captured_color, CV_GRAY2BGR);
	drawChessboardCorners(captured_color, patternSize, cv::Mat(corners), found);
	cv::imshow("Captured_color", captured_color);

	cv::waitKey(1);

	if (!found) {
		return -1;
	}

	//double energy = energyAll(corners, patternSize, capturedDistCenter_est.x, capturedDistCenter_est.y);
	double energy = energyAll(corners, patternSize);

	return energy;
}

double mOptimizer::SA_F(cv::Mat & proj_pattern, cv::Size & patternSize, std::vector<double> k1k2_w)
{
	return  SA_F( proj_pattern,  patternSize, k1k2_w[0], k1k2_w[1]);
}

void mOptimizer::findDistCenter(cv::Mat & proj_pattern, cv::Size & patternSize, cv::Point2d & center)
{
	cv::Mat pattern = proj_pattern.clone();
	cv::Mat pattern_color;
	cv::cvtColor(pattern, pattern_color, CV_GRAY2BGR);
	cv::Mat distored,captured;
	cv::Mat corners;


	cv::circle(pattern_color, distCenter_gt, 10, cv::Scalar(0, 0,255), 2);
	imshow("pattern_color", pattern_color);
	cv::waitKey();

	//simulate projecting and capturing
	capture_simulate(pattern, distored, captured, map_x_p, map_y_p);

	imshow("distored", distored);
	imshow("captured", captured);
	cv::waitKey();

	//extract corner pixels from captured image
	extractCorners(captured, patternSize, corners);
	cv::Mat temp;
	cv::cvtColor(captured, temp, CV_GRAY2BGR);
	cv::drawChessboardCorners(temp, patternSize, corners, true);
	imshow("temp", temp);

	//testing. find the initial homography
	hg = cv::findHomography(corners, src_corners);
	cv::Mat warped;
	cv::warpPerspective(captured, warped, hg,cv::Size(1024,768));
	cv::imshow("warped", warped);
	cv::waitKey();

	
	//first round. to find the row/col of pattern which is nearest to the distortion center
	std::vector<cv::Mat> lines_col;
	extractLines(corners, lines_col, patternSize, ext_line_col);
	int min_col = minEnergyLine(lines_col);
	std::cout << "min_col= " << min_col << std::endl;

	std::vector<cv::Mat> lines_row;
	extractLines(corners, lines_row, patternSize, ext_line_row);
	int min_row = minEnergyLine(lines_row);
	std::cout << "min_row= " << min_row << std::endl;

	cv::waitKey();
	//loop: shift the original pattern. find the offset which locate min_row, min_col to the distortion center.
	int grid_size = 64;
	cv::Mat shifted, shifted_alt;
	double min_energy_row = std::numeric_limits<double>::max();
	double min_energy_col = std::numeric_limits<double>::max();
	double  min_offset_x, min_offset_y;
	int grid_offset_col = 0;
	for (int i = -grid_size / 2; i <= grid_size / 2; i++) {
		//shift horizontally
		translateImg(pattern, shifted, i, 0);
		//translateImg(pattern, shifted_alt, i, grid_size/2.0);
		imshow("shifted", shifted);
		//imshow("shifted_alt", shifted_alt);

		//simulate projecting and capturing
		cv::Mat distored_alt, captured_alt;
		capture_simulate(shifted, distored, captured, map_x_p, map_y_p);
		imshow("captured", captured);
		//capture_simulate(shifted_alt, distored_alt, captured_alt, map_x_p, map_y_p);
		//imshow("captured_alt", captured_alt);

		bool found = extractCorners(captured, patternSize, corners);
		cv::Mat corners_alt;
		//bool found_alt = extractCorners(captured_alt, patternSize, corners_alt);
		if (found ) {
			//std::cout << corners << std::endl;

			extractLines(corners, lines_col, patternSize, ext_line_col);
			//std::vector<cv::Mat> lines_col_alt;
			//extractLines(corners_alt, lines_col_alt, patternSize, ext_line_col);
			cv::Mat line = lines_col[min_col];
			//cv::hconcat(lines_col[min_col], lines_col_alt[min_col], line);
			//double e = energy(lines_col[min_col]);
			double e = energy(line);
			if (e >= 0 && e < min_energy_col) {
				min_energy_col = e;
				min_offset_x = i;
				grid_offset_col = 0;
			}
			if (min_col - 1 >= 0) {
				line = lines_col[min_col - 1];
				//cv::hconcat(lines_col[min_col-1], lines_col_alt[min_col-1], line);
				e = energy(line);
				if (e >= 0 && e < min_energy_col) {
					min_energy_col = e;
					min_offset_x = i;
					grid_offset_col = -1;
				}
			}
			if (min_col + 1 < patternSize.width) {
				line = lines_col[min_col + 1];
				//cv::hconcat(lines_col[min_col+1], lines_col_alt[min_col+1], line);
				e = energy(line);
				if (e >= 0 && e < min_energy_col) {
					min_energy_col = e;
					min_offset_x = i;
					grid_offset_col = 1;
				}
			}
		}
		else {
			std::cout << "fail" << std::endl;
		}

		cv::waitKey(10);
	}
	min_offset_x = min_offset_x + grid_offset_col * grid_size;

	int grid_offset_row = 0;
	for (int j = -grid_size / 2; j <= grid_size / 2; j++) {
		//shift vertically
		translateImg(pattern, shifted, 0, j);
		//translateImg(pattern, shifted_alt, grid_size / 2.0, j);
		imshow("shifted", shifted);
		//imshow("shifted_alt", shifted_alt);

		//simulate projecting and capturing
		cv::Mat distored_alt, captured_alt;
		capture_simulate(shifted, distored, captured, map_x_p, map_y_p);
		imshow("captured", captured);
		//capture_simulate(shifted_alt, distored_alt, captured_alt, map_x_p, map_y_p);
		//imshow("captured_alt", captured_alt);

		bool found = extractCorners(captured, patternSize, corners);
		cv::Mat corners_alt;
		//bool found_alt = extractCorners(captured_alt, patternSize, corners_alt);
		if (found) {

			extractLines(corners, lines_row, patternSize, ext_line_row);
			//std::vector<cv::Mat> lines_row_alt;
			//extractLines(corners_alt, lines_row_alt, patternSize, ext_line_row);
			cv::Mat line = lines_row[min_row];
			//cv::hconcat(lines_row[min_row], lines_row_alt[min_row], line);
			double e = energy(line);
			if (e >= 0 && e < min_energy_row) {
				min_energy_row = e;
				min_offset_y = j;
				grid_offset_row = 0;
			}
			if (min_row - 1 >= 0) {
				line = lines_row[min_row - 1];
				//cv::hconcat(lines_row[min_row-1], lines_row_alt[min_row-1], line);
				e = energy(line);
				if (e >= 0 && e < min_energy_row) {
					min_energy_row = e;
					min_offset_y = j;
					grid_offset_row = -1;
				}
			}
			if (min_row + 1 < patternSize.height) {
				line = lines_row[min_row + 1];
				//cv::hconcat(lines_row[min_row+1], lines_row_alt[min_row+1], line);
				e = energy(line);
				if (e >= 0 && e < min_energy_row) {
					min_energy_row = e;
					min_offset_y = j;
					grid_offset_row = 1;
				}
			}
		}
		else {
			std::cout << "fail" << std::endl;
		}

		cv::waitKey(10);
	}
	min_offset_y = min_offset_y + grid_offset_row * grid_size;

	//convert index to coordinate
	cv::Mat corners_src;
	extractCorners(pattern, patternSize,corners_src);
	double x = corners_src.at<cv::Point2f>(min_col).x;
	double y = corners_src.at<cv::Point2f>(patternSize.width*min_row).y;

	center.x = x + min_offset_x;
	center.y = y + min_offset_y;
	distCenter_est.x = center.x;
	distCenter_est.y = center.y;
	toCapturedCoordinate(distCenter_est, capturedDistCenter_est);

	cv::circle(pattern_color, center, 10, cv::Scalar(100, 200, 0),2);
	imshow("pattern_color", pattern_color);
	cv::waitKey();

	std::cout << "x = " << x << std::endl;
	std::cout << "min_offset_x = " << min_offset_x << std::endl;
 	std::cout <<"dist center = "<< center << std::endl;

}

void mOptimizer::init(double k1_p, double k2_p, cv::Size size)
{
	std::cout << "Initializing..." << std::endl;
	map_x_p.create(size, CV_32FC1);
	map_y_p.create(size, CV_32FC1);
	distCenter_gt = cv::Point2d(200.5,580.5);
	//distCenter_gt = cv::Point2d(500.5, 380.5);
	//createMap(map_x_p, map_y_p, k1_p, k2_p, 1024/2.f, 768/2.f);
	createMap(map_x_p, map_y_p, k1_p, k2_p, distCenter_gt.x, distCenter_gt.y);

	readHomography();
}
