cmake_minimum_required(VERSION 2.8.3)
project(linefeature_detection)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    message_filters
   # message_generation
    geometry_msgs
    line_msgs
    )

find_package(OpenCV REQUIRED)

#add_message_files(
#FILES
#lines2d.msg
#)

#generate_messages(
#DEPENDENCIES 
#std_msgs
#)


catkin_package(
#CATKIN_DEPENDS
#message_runtimes
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3  REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)


add_executable(linefeature_detection
    src/linefeature_node.cpp
    src/parameters.cpp
    src/linefeature.cpp
    )

add_dependencies(linefeature_detection
line_msgs_gencpp
)

target_link_libraries(linefeature_detection ${catkin_LIBRARIES} ${OpenCV_LIBS})






