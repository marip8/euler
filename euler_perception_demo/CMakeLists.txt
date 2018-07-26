cmake_minimum_required(VERSION 2.8.3)
project(euler_perception_demo)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  eigen_stl_containers
  eigen_conversions
  pcl_ros
  std_srvs
  message_generation
  moveit_ros_planning_interface
)

add_service_files(
  FILES
    LocalizePart.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    sensor_msgs
    eigen_stl_containers
    eigen_conversions
    pcl_ros
    std_srvs
    message_runtime
    moveit_ros_planning_interface
)

###########
## Build ##
###########

include_directories(
 ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node
  src/perception_demo.cpp
)
add_dependencies(${PROJECT_NAME}_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)

add_executable(execution_node
  src/execution_node.cpp
)
add_dependencies(execution_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(execution_node
  ${catkin_LIBRARIES}
)
