/**
 * CVSensorSimulator tracks the pose of objects fitted with AprilTags in view of
 * an overhead camera and sends that pose data to microUSV's over TCP.
 * 
 * Modified Robot class to implement the Lasso method on the Kodama Robots.
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

#include "Robot.h"

using namespace std;
using namespace cv;

Robot::Robot(int tagID, string label, int img_width, int img_height) {
	this->tagID = tagID;
	this->label = label;
	this->communication_range = 600;

	this->x_res = img_width;
	this->y_res = img_height;

	tagRGB = make_tuple(0, 0, 255);
	gettimeofday(&this->pose.timestamp, NULL);
}

Robot::~Robot() {
}

void Robot::updateSensorValues(Mat targets, Mat scalarField, vector<pose2D> Robot_poses, int my_index) {
	sensorVals_incomplete.pose = Robot_poses.at(my_index);
	
	int col = (int) round(sensorVals_incomplete.pose.x);
	int row = (int) round(sensorVals_incomplete.pose.y);
	sensorVals_incomplete.centreGridSensor = scalarField.at<uchar>(row, col) / 255.0;
	
	sensorVals_incomplete.nearbyVesselPoses.clear();
	for (int i = 0; i < Robot_poses.size(); i++) {
		if(i != my_index) {
			double range = getTargetRange(Robot_poses.at(my_index), Robot_poses.at(i));
			if(range < communication_range) {
				sensorVals_incomplete.nearbyVesselPoses.push_back(Robot_poses.at(i));
			}
		}
	}

	sensorVals_incomplete.highestVisiblePuckValue = 42;

	std::lock_guard<std::mutex> lock(sensorVal_lock);
	sensorVals_complete = sensorVals_incomplete;
}

SensorValues Robot::getSensorValues() {
	std::lock_guard<std::mutex> lock(sensorVal_lock);
	return sensorVals_complete;
}

double Robot::getTargetRange(pose2D my_pose, pose2D target_pose) {
	return sqrt(pow(my_pose.x - target_pose.x,2.0) + pow(my_pose.y - target_pose.y,2.0));
}

double Robot::getTargetHeading(pose2D my_pose, pose2D target_pose) {
	return atan2(target_pose.x - my_pose.x, -(target_pose.y - my_pose.y));
}