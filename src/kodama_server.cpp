/**
 * Modified version of CVSensorSimulator to implement the Lasso method on the Kodama robots.
 *
 * Copyright (C) 2019  CalvinGregory  cgregory@mun.ca
 * 				 2021  Andrew Vardy	  av@mun.ca
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
 */

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <unistd.h>
#include <netinet/in.h>
#include <thread>
#include <functional>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>
#include <chrono>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <google/protobuf/util/time_util.h>
#include <sys/stat.h>

#include "FrameBuffer.h"
#include "PoseDetector.h"
#include "Robot.h"
#include "ConfigParser.h"
#include "CVSS_util.h"
#include "cyclicbarrier.hpp"
#include "CSVWriter.h"
#include "kodama_msg.pb.h"

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag25h9.h"
#include "apriltag/apriltag_pose.h"
}

using namespace std;
using namespace cv;
using google::protobuf::util::TimeUtil;

#define PORT 8078

bool running;
bool visualize;
ConfigParser::Config config;
Mat frame;
Mat labelledDetections;
Mat targetMask;
Mat displayFrame;
std::mutex displayFrameMutex;

// BAD: Hard-coded goal position.
Point goalPosition{310, 245};

// Images that will be loaded from local files.
Mat freeSpace, scalarField;//, goal;

// The last request from a client.  Made it global so we can use it for
// visualization.
kodama::RequestData requestData;

// Threshold on cluster area to count as a target.
int areaThreshold = 150;

// Build global barriers
class concrete_callable : public cbar::callable {
public:
	concrete_callable() {}
	virtual void run() override {}
};
auto cc = new concrete_callable();
cbar::cyclicbarrier* frameAcquisitionBarrier = new cbar::cyclicbarrier(2,cc);
cbar::cyclicbarrier* detectorBarrier0 = new cbar::cyclicbarrier(3,cc);
cbar::cyclicbarrier* detectorBarrier1 = new cbar::cyclicbarrier(3,cc);

/*
 This function is removed because all UI operations, including waitKey,
 must be performed on the main thread to comply with macOS requirements.
void afterImshow() {
    if (waitKey(1) == 27) {
        running = false;
    }
}
*/

/**
 * Video Capture thread function. Continuously updates the FrameBuffer.
 *
 * @param fb The FrameBuffer object to update.
 */
void video_capture_thread(FrameBuffer& fb) {
	while(running) {
		fb.updateFrame();
	}
}

/**
 * AprilTag detector thread function. Detects poses of AprilTags in the most
 * recent frame data from the FrameBuffer. Generates visualization video feed.
 *
 * @param pd The PoseDetector object which performs detections.
 * @param fb The FrameBuffer object to query for new camera frames.
 */
void apriltag_detector_thread(PoseDetector& pd, FrameBuffer& fb) {
	Mat apriltag_frame(100, 100, CV_8UC3, Scalar(0,0,0));
	// imshow("apriltag_detector_thread - 1", apriltag_frame);
	while (running) {
		frame = fb.getFrame();
		apriltag_frame = frame;
		frameAcquisitionBarrier->await();

		if (!apriltag_frame.empty()) {
			pd.updatePoseEstimates(&apriltag_frame); 
			if(visualize) {
				labelledDetections = *pd.getLabelledFrame(config);
				// imshow("apriltag_detector_thread - 2", labelledDetections);
				// afterImshow();
			}
		}
		
		detectorBarrier0->await();
		detectorBarrier1->await();
	}
}

/**
 * Target detector thread function. Detects colored targets in the camera frame. 
 * Saves a global mask image indicating which pixels are the targeted color. 
 */
void target_detector_thread() {
	Mat hsv;
	//int dead_zone_thickness = 75;
	//Mat dead_zone_mask(720, 1280, CV_8U, Scalar(0,0,0));
	//rectangle(dead_zone_mask, Point(dead_zone_thickness, dead_zone_thickness), Point(1280 - dead_zone_thickness, 720 - dead_zone_thickness), Scalar(255,255,255), CV_FILLED);

	while (running) {
		frameAcquisitionBarrier->await();
		frameAcquisitionBarrier->reset();
		try {
			cvtColor(frame, hsv, COLOR_BGR2HSV);
			cv::inRange(hsv, config.target_thresh_low, config.target_thresh_high, targetMask);
			medianBlur(targetMask, targetMask, 7);

			bitwise_and(freeSpace, targetMask, targetMask);
			
			//DEBUG
			// imshow("target_detector_thread", targetMask);
			// afterImshow();
		} 
		catch(cv::Exception) {} // Handles 0.56% chance cvtColor will throw cv::Exception error due to empty frame
		
		detectorBarrier0->await();
		detectorBarrier1->await();
	}
}

/**
 * Thread function to process apriltag and target detections data. Generates simulated 
 * sensor values for target and obstacle detections to be sent to each microUSV and 
 * can export each vessel's pose history to a CSV file. 
 * 
 * @param config Configuration data extracted from provided json file. Contains camera information.
 * @param robots List of all robots being tracked by the simulator.
 * @param csv List of CSV file objects recording the pose of each robot.
 * @param output_csv Flag indicating if robot pose data should be recorded to the csv files. 
 */
void detection_processor_thread(ConfigParser::Config& config, vector<shared_ptr<Robot>>& robots, vector<CSVWriter>& vessel_pose_csv, vector<CSVWriter>& target_pose_csv, bool output_csv) {
    auto start_time = chrono::steady_clock::now();
    Mat targets;
    Scalar ContourColor(255, 0, 0);
    Scalar TargetMarkerColor(255, 200, 200);
    Scalar PuckMarkerColor(0, 255 ,0);
    Scalar GoalColor(0, 0, 255);

    while (running) {
        detectorBarrier0->await();
        detectorBarrier0->reset();
        
        auto current_time = chrono::steady_clock::now();
        targets = targetMask.clone();
        
        vector<pose2D> robot_poses;
        for (int i = 0; i < robots.size(); i++) {
            robot_poses.push_back(robots.at(i)->getPose());
        }
        
        detectorBarrier1->await(); 
        detectorBarrier1->reset();

        // Detect all targets
        Mat labelImage(targetMask.size(), CV_32S);
        Mat stats, centroids;
        int nLabels = connectedComponentsWithStats(targets, labelImage, stats, centroids, 4, CV_32S);
        vector<position2D> allTargets;
        for (int i = 1; i < nLabels; i++) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) > areaThreshold) {
                int x = cvRound(centroids.at<double>(i, 0));
                int y = cvRound(centroids.at<double>(i, 1));
                allTargets.push_back(position2D(x, y));
            }
        }
        
        for (int i = 0; i < robots.size(); i++)
            robots.at(i)->updateSensorValues(allTargets, robot_poses, i);

        if(visualize || output_csv) {

            if (visualize) {
                Mat localDisplayFrame = labelledDetections.clone();

                // Create a BGR version of targetMask for display purposes.
                Mat targetMaskBGR;
                cvtColor(targetMask, targetMaskBGR, COLOR_GRAY2BGR);

                // This will highlight pixels in the target colour in white.
                bitwise_or(targetMaskBGR, localDisplayFrame, localDisplayFrame);

                for(int i = 0; i < allTargets.size(); i++) {
                    position2D & p = allTargets.at(i);
                    circle(localDisplayFrame, Point(p.x_px, p.y_px), 12, PuckMarkerColor, 2);
                }

                // Display the contour from the last connected robot.
                double tau = requestData.tau() / 1000.0;

                // CAN'T GET THIS TO WORK: OpenCV's contour-finding
                /*
                std::vector<std::vector<Point>> contours;
                findContours(scalarField, contours, RETR_LIST, CHAIN_APPROX_NONE);
                drawContours(displayFrame, contours, 0, TargetMarkerColor); 
                */
            /*
                for (int j=0; j < scalarField.rows; ++j)
                    for (int i=0; i < scalarField.cols; ++i)
                        if (abs(scalarField.at<uchar>(j, i) / 255.0 - tau) < 0.01) {
                            circle(displayFrame, Point{i, j}, 1, ContourColor, 1);
                        }
            */

                // Display the target position from the last connected robot.
                circle(localDisplayFrame, Point{requestData.targetx(), requestData.targety()}, 5, TargetMarkerColor, 1);

                // Display the goal position as an X.
                line(localDisplayFrame, goalPosition + Point{-20, -20}, goalPosition + Point{20, 20}, GoalColor, 3);
                line(localDisplayFrame, goalPosition + Point{-20, 20}, goalPosition + Point{20, -20}, GoalColor, 3);

                // Safely update the global display frame for the main thread to render.
                {
                    std::lock_guard<std::mutex> lock(displayFrameMutex);
                    displayFrame = localDisplayFrame.clone();
                }
            }

            if (output_csv) {
                double timestamp = chrono::duration_cast<chrono::nanoseconds>(current_time - start_time).count();
                for(uint i = 0; i < vessel_pose_csv.size(); i++) {
                    vessel_pose_csv.at(i).newRow() << timestamp << robot_poses.at(i).x << robot_poses.at(i).y << robot_poses.at(i).yaw;
                }
                target_pose_csv.at(0).newRow() << timestamp << allTargets.size();
                for(uint i = 0; i < allTargets.size(); i++) {
                    position2D &target = allTargets.at(i);
                    target_pose_csv.at(0) << "" << target.x_px << target.y_px;
                }

                int nRows = targets.rows;
                int nCols = targets.cols * targets.channels();
                if (targets.isContinuous()) {
                    nCols *= nRows;
                    nRows = 1;
                }
                uint px_count = 0;
                double px_distance_sum = 0;
                uchar* px;
                for(uint i = 0; i < targets.rows; i++) {
                    px = targets.ptr<uchar>(i);
                    for(uint j = 0; j < targets.cols * targets.channels(); j++) {
                        if(px[j] > 0) {
                            px_count++;
                            px_distance_sum += sqrt(pow(config.cInfo.cx - j, 2) + pow(config.cInfo.cy - i, 2));
                        }
                    }
                }
                double average_px_distance = 0;
                if (px_count > 0) {
                    average_px_distance = px_distance_sum/px_count/targets.channels();
                }
                target_pose_csv.at(1).newRow() << timestamp << px_count << average_px_distance;
            }
        }		
    }

	// Cleanup barrier objects
	delete cc;
	delete frameAcquisitionBarrier;
	delete detectorBarrier0;
	delete detectorBarrier1;
}

int main(int argc, char* argv[]) {
//--- Initialize Threads ---//
	running = true;
	struct timeval startTime;
	gettimeofday(&startTime, NULL);

	// Parse config file and pull values into local variables.
	if (argc > 1) {
		config = ConfigParser::getConfigs(argv[1]);
	}
	// If no file path provided attempt to open default file name. 
	else {
		config = ConfigParser::getConfigs("config.json");
	}
	
	visualize = config.visualize;

	apriltag_detection_info_t info;

	info.tagsize = config.tagsize;
	info.fx = config.cInfo.fx;
	info.fy = config.cInfo.fy;
	info.cx = config.cInfo.cx;
	info.cy = config.cInfo.cy;

	VidCapSettings settings;

	settings.cameraID = config.cInfo.cameraID;
	settings.x_res = config.cInfo.x_res;
	settings.y_res = config.cInfo.y_res;

	int size = config.robots.size(); 
	vector<shared_ptr<Robot>> robots(config.robots);
	vector<CSVWriter> vessel_pose_csv(config.robots.size());
	vector<CSVWriter> target_pose_csv(2);

	if (config.output_csv) {
		for (uint i = 0; i < vessel_pose_csv.size(); i++) {
			vessel_pose_csv.at(i).newRow() << "Timestamp [ns]" << "X [mm]" << "Y [mm]" << "Yaw [rad]";
		}
		target_pose_csv.at(0).newRow() << "Timestamp [ns]" << "Number of Targets" << "" << "Target Positions X & Y [px]" << "" << "" << "etc.";
		target_pose_csv.at(1).newRow() << "Timestamp [ns]" << "Number of Target Pixels" << "Average target pixel distance to cluster point";
	}

	// Load images.  Make the goal image contain red only.
	Mat obstacles = imread("obstacles.png", IMREAD_GRAYSCALE);
	bitwise_not(obstacles, freeSpace);
	scalarField = imread("travel_time.png", IMREAD_GRAYSCALE);
	//bitwise_not(imread("../images/live/stadium_one_wall/goal.png", IMREAD_COLOR), goal);

	// This is done just so that a contour won't be shown if there are no robots responding yet.
	requestData.set_tau(-1);

	// Build thread parameter objects.
	FrameBuffer fb(settings);
	PoseDetector pd(info, robots);
	Mat targetMask;

	// Start threads
	thread threads[4];
	threads[0] = thread(video_capture_thread, ref(fb));
	threads[1] = thread(apriltag_detector_thread, ref(pd), ref(fb));
	threads[2] = thread(target_detector_thread);
	threads[3] = thread(detection_processor_thread, ref(config), ref(robots), ref(vessel_pose_csv), ref(target_pose_csv), config.output_csv);

	for (int i = 0; i < 4; i++) {
		threads[i].detach();
	}

//--- Socket Setup --- //
	int server_fd, new_socket;
	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);
	char buffer[256] = {0};

	// Creating socket file descriptor
	if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
	{
		perror("socket failed");
		exit(EXIT_FAILURE);
	}

	// Forcefully attach socket to the port
	if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
	{
		perror("setsockopt");
		exit(EXIT_FAILURE);
	}
	// Make socket non-blocking
	fcntl(server_fd, F_SETFL, O_NONBLOCK);

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons( PORT );

	// Forcefully attach socket to the port
	if (::bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
	{
		perror("bind failed");
		exit(EXIT_FAILURE);
	}
	if (listen(server_fd, 10) < 0)
	{
		perror("listen");
		exit(EXIT_FAILURE);
	}

//--- Request Receiver Loop ---//
	struct timeval currentTime;

	while (running) {
		//kodama::RequestData requestData;
		kodama::SensorData sensorData;

		// Connect to Kodama and receive data.
		new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
		if(new_socket == -1) {
			if (errno != EWOULDBLOCK) {
				perror("error accepting connection");
				exit(EXIT_FAILURE);
			}
		}
		else {
			read(new_socket, buffer, 128);
			if (!requestData.ParseFromString(buffer)) {
				cerr << "Failed to parse Request Data message." << endl;
				cerr << "buffer: " << buffer << endl;
				return -1;
			}

			// Identify microUSV and respond with its sensor data.
			int index = CVSS_util::tagMatch(robots, requestData.tag_id());
			SensorValues sensorValues = robots[index]->getSensorValues();

			currentTime = sensorValues.pose.timestamp;
			long seconds = currentTime.tv_sec - startTime.tv_sec;
			long uSeconds = currentTime.tv_usec - startTime.tv_usec;
			if (uSeconds < 0) {
				uSeconds = uSeconds + 1e6;
				seconds--;
			}

			sensorData.mutable_pose()->set_x(sensorValues.pose.x);
			sensorData.mutable_pose()->set_y(sensorValues.pose.y);
			sensorData.mutable_pose()->set_yaw(sensorValues.pose.yaw);
			//sensorData.mutable_pose()->set_xpx(sensorValues.pose.x_px);
			//sensorData.mutable_pose()->set_ypx(sensorValues.pose.y_px);
			for(int i = 0; i < sensorValues.nearbyRobotPoses.size(); i++) {
				kodama::SensorData_Pose2D* nearbyVessel = sensorData.add_nearby_robot_poses();
				nearbyVessel->set_x(sensorValues.nearbyRobotPoses.at(i).x);
				nearbyVessel->set_y(sensorValues.nearbyRobotPoses.at(i).y);
				nearbyVessel->set_yaw(sensorValues.nearbyRobotPoses.at(i).yaw);
				//nearbyVessel->set_xpx(sensorValues.nearbyVesselPoses.at(i).x_px);
				//nearbyVessel->set_ypx(sensorValues.nearbyVesselPoses.at(i).y_px);
			}
			for(int i = 0; i < sensorValues.nearbyTargetPositions.size(); i++) {
				kodama::SensorData_Position2D* nearbyTarget = sensorData.add_nearby_target_positions();
				nearbyTarget->set_x(sensorValues.nearbyTargetPositions.at(i).x_px);
				nearbyTarget->set_y(sensorValues.nearbyTargetPositions.at(i).y_px);
				//nearbyVessel->set_xpx(sensorValues.nearbyVesselPoses.at(i).x_px);
				//nearbyVessel->set_ypx(sensorValues.nearbyVesselPoses.at(i).y_px);
			}
			*sensorData.mutable_timestamp() = TimeUtil::MicrosecondsToTimestamp(seconds * 1e6 + uSeconds);

			size_t size = sensorData.ByteSizeLong();
//std::cerr << "SensorData size (bytes): " << size << endl;
			char* msg = new char [size];
			sensorData.SerializeToArray(msg, size);

			send(new_socket, msg, size, 0);
			close(new_socket);
			std::memset(buffer,0,128);
			delete[] msg;
		}

        if (visualize) {
            Mat frameToShow;
            {
                std::lock_guard<std::mutex> lock(displayFrameMutex);
                if (!displayFrame.empty()) {
                    frameToShow = displayFrame.clone();
                }
            }
            if (!frameToShow.empty()) {
                imshow("Kodama Server", frameToShow);
            }
            // Process UI events and check for exit key (ESC)
            if (waitKey(1) == 27) {
                running = false;
            }
        } else {
            // If not visualizing, prevent the loop from consuming 100% CPU.
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }

    destroyAllWindows();

    if (config.output_csv) {
        auto t = std::time(nullptr);
        auto tm = *localtime(&t);
        stringstream dirName;
        dirName << "cvss_data_";
        dirName << put_time(&tm, "%Y-%m-%d_%H:%M");
        mkdir(dirName.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        for (uint i = 0; i < vessel_pose_csv.size(); i++) {
            stringstream fileName;
            fileName << dirName.str();
            fileName << "/";
            fileName << robots[i]->getLabel();
            fileName << "_pose_data_";
            fileName << put_time(&tm, "%Y-%m-%d_%H:%M");
            fileName << ".csv";
            vessel_pose_csv[i].writeToFile(fileName.str());
        }
        {
            stringstream fileName;
            fileName << dirName.str();
            fileName << "/";
            fileName << "Target_position_data_";
            fileName << put_time(&tm, "%Y-%m-%d_%H:%M");
            fileName << ".csv";
            target_pose_csv.at(0).writeToFile(fileName.str());
        }
        {
            stringstream fileName;
            fileName << dirName.str();
            fileName << "/";
            fileName << "Target_pixel_distance_data_";
            fileName << put_time(&tm, "%Y-%m-%d_%H:%M");
            fileName << ".csv";
            target_pose_csv.at(1).writeToFile(fileName.str());
        }
    }
	
	return 0;
}
