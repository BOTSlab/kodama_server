/**
 * Modified Robot class to implement the Lasso method on the Kodama robots.
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

#ifndef ROBOT_H_
#define ROBOT_H_


#include "TaggedObject.h"
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"

struct position2D {
	position2D(uint x, uint y)
		: x_px{x}, y_px{y}
	{}
	uint x_px = 0;
	uint y_px = 0;
};

/**
 * SensorValues is a struct containing all data contained in a SensorData message. 
 */ 
typedef struct {
	pose2D pose;
	std::vector<position2D> nearbyTargetPositions;
	std::vector<pose2D> nearbyRobotPoses;
} SensorValues;

/**
 * The Robot class represents a Kodama robot marked with an AprilTag.
 */
class Robot : public TaggedObject {
protected:
	int x_res;
	int y_res;
	double robot_sensing_distance;
	double target_sensing_distance;
	std::mutex sensorVal_lock;
	SensorValues sensorVals_complete;
	SensorValues sensorVals_incomplete;
	double getTargetRange(pose2D my_pose, pose2D target_pose);
	double getTargetHeading(pose2D my_pose, pose2D target_pose);
public:
	/**
	 * @param tagID This Robot's tagID.
	 * @param label This Robot's label string.
	 * @param img_width Camera frame x dimension in pixels.
	 * @param img_height Camera frame y dimension in pixels
	 */
	Robot(int tagID, std::string label, int img_width, int img_height);
	~Robot();
	/**
	 * Updates the SensorValues stored in this Robot based on the provided list of Robot poses and target mask.
	 * 
	 * @param allTargets All targets detected in the camera frame.
	 * @param Robot_poses Vector of poses for each Robot in the system.
	 * @param my_index The index of this Robot in the Robot_poses vector.
	 */
	void updateSensorValues(std::vector<position2D> allTargets, std::vector<pose2D> Robot_poses, int my_index);
	/**
	 * Retrieves the most recent SensorValues stored in this Robot. This function is thread safe.
	 * 
	 * @return A SensorValues struct populated with the most up to date data. 
	 */
	SensorValues getSensorValues();
};


#endif /* ROBOT_H_ */
