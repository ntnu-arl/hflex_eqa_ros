cmake_minimum_required(VERSION 3.22.1)
project(simulation_manager_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(hflex_eqa_msgs REQUIRED)

install(PROGRAMS app/habitat_eqa_manager_node.py
        DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})
ament_package()
