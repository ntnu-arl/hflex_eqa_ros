cmake_minimum_required(VERSION 3.22.1)
project(hflex_eqa_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(config_utilities REQUIRED)
find_package(config_utilities_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(hydra REQUIRED)
find_package(hydra_msgs REQUIRED)
find_package(hflex_eqa REQUIRED)
find_package(hflex_eqa_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ianvs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(spark_dsg REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(semantic_inference_msgs REQUIRED)
find_package(vlm_msgs REQUIRED)

add_library(
  ${PROJECT_NAME}
  src/hflex_eqa_ros_pipeline.cpp
  src/input/input_receiver.cpp
  src/io/graph_file_wrapper.cpp
  src/io/graph_zmq_wrapper.cpp
  src/io/graph_ros_wrapper.cpp
  src/low_level/low_level_publisher.cpp
  src/search_manager/manager_ros_interface.cpp
  src/search_manager/monitor_publisher.cpp
  src/search_manager/search_manager_publisher.cpp
  src/utils/conversions.cpp
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC config_utilities::config_utilities
         config_utilities_ros::config_utilities_ros
         hydra::hydra
         hflex_eqa::hflex_eqa
         ianvs::ianvs
         rclcpp::rclcpp
         spark_dsg::spark_dsg
         tf2_eigen::tf2_eigen
         tf2_ros::tf2_ros
  PRIVATE cv_bridge::cv_bridge glog::glog
)
ament_target_dependencies(
  ${PROJECT_NAME}
  PUBLIC
  hydra_msgs
  nav_msgs
  semantic_inference_msgs
  vlm_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  hflex_eqa_msgs
)

add_executable(hflex_eqa_ros_node app/hflex_eqa_node.cpp)

target_link_libraries(
  hflex_eqa_ros_node
  -Wl,--whole-archive
  hflex_eqa::hflex_eqa
  -Wl,--no-whole-archive
  ${PROJECT_NAME}
  gflags
  glog::glog
)

# TODO(albert): add abnd enable tests
# if(BUILD_TESTING)
#   find_package(ament_cmake_gtest)
#   ament_add_gtest(test_${PROJECT_NAME} tests/main.cpp)
#   target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} gflags)
# endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)


install(PROGRAMS app/vlms_node.py
                 hflex_eqa_ros/scripts/habitat_image_feature_saver_node.py
                 test/test_tsdf_node.py
        DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS hflex_eqa_ros_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

ament_export_targets(${PROJECT_NAME}-targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  config_utilities
  config_utilities_ros
  hydra
  hflex_eqa
  hydra_msgs
  ianvs
  rclcpp
  spark_dsg
  sensor_msgs
  std_msgs
  std_srvs
  semantic_inference_msgs
  vlm_msgs
  tf2_eigen
  tf2_ros
  nav_msgs
  visualization_msgs
  geometry_msgs
  hflex_eqa_msgs
)
ament_python_install_package(${PROJECT_NAME})
ament_package()
