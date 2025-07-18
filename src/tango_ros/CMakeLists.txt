cmake_minimum_required(VERSION 2.8.3)
project(tango_ros)

add_compile_options(-std=c++11)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  message_generation
  dynamic_reconfigure
  sensor_msgs
  tf
  cv_bridge
)

find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIR})
    message(STATUS "OpenCV ${OpenCV_VERSION} FOUND")
    message(STATUS ${OpenCV_LIBS})
    if(${OpenCV_VERSION_MAJOR} EQUAL 4)
        add_definitions(-DOPENCV_VERSION_4)
    elseif(${OpenCV_VERSION_MAJOR} LESS 4)
        add_definitions(-DOPENCV_VERSION_3)
    endif()
else()
    message(FATAL_ERROR "OpenCV Library Not Found")
endif()

# catkin_python_setup()

# Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/Interface.cfg
# )

include_directories(
    include
    ${PROJECT_SOURCE_DIR}/msg
    ${PROJECT_SOURCE_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_service_files(
  FILES
)

#add_message_files(
#  DIRECTORY msg
#  FILES
#)

# generate_messages(
#  DEPENDENCIES
#  sensor_msgs
#  std_msgs
#  geometry_msgs
#  nav_msgs
#  tf
#)

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()

link_directories(lib/cpp/${ARCH})

set(CMAKE_CXX_FLAGS "-O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

catkin_package(
   CATKIN_DEPENDS roscpp rospy message_runtime rospy sensor_msgs nav_msgs tf dynamic_reconfigure cv_bridge
)

# install
install(DIRECTORY
  config
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
