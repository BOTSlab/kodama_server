cmake_minimum_required(VERSION 3.10)
project(kodama_server C CXX)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui features2d calib3d videoio imgcodecs imgproc)
find_package(Protobuf CONFIG REQUIRED)

# Define AprilTag C source files
set(APRILTAG_SOURCES
    src/apriltag/apriltag.c
    src/apriltag/apriltag_pose.c
    src/apriltag/apriltag_quad_thresh.c
    src/apriltag/tag25h9.c
    src/apriltag/tag36h11.c
    src/apriltag/common/g2d.c
    src/apriltag/common/getopt.c
    src/apriltag/common/homography.c
    src/apriltag/common/image_u8.c
    src/apriltag/common/image_u8x3.c
    src/apriltag/common/image_u8x4.c
    src/apriltag/common/matd.c
    src/apriltag/common/pam.c
    src/apriltag/common/pjpeg-idct.c
    src/apriltag/common/pjpeg.c
    src/apriltag/common/pnm.c
    src/apriltag/common/string_util.c
    src/apriltag/common/svd22.c
    src/apriltag/common/time_util.c
    src/apriltag/common/unionfind.c
    src/apriltag/common/workerpool.c
    src/apriltag/common/zarray.c
    src/apriltag/common/zhash.c
    src/apriltag/common/zmaxheap.c
)

# Define application C++ source files
set(APP_SOURCES
    src/ConfigParser.cpp
    src/CVSS_util.cpp
    src/kodama_server.cpp
    src/cyclicbarrier.cpp
    src/FrameBuffer.cpp
    src/kodama_msg.pb.cc
    src/PoseDetector.cpp
    src/Robot.cpp
)

# Create the executable from all source files
add_executable(kodama_server ${APP_SOURCES} ${APRILTAG_SOURCES})

# Add include directories for headers
target_include_directories(kodama_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/apriltag
    ${OpenCV_INCLUDE_DIRS}
)

# Link against required libraries
target_link_libraries(kodama_server PRIVATE
    Threads::Threads
    ${OpenCV_LIBS}
    protobuf::libprotobuf
)