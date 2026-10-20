cmake_minimum_required(VERSION 3.16)

project(gtsam_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTSAM 4.3 REQUIRED)

add_executable(
  gtsam_reference
  gtsam_reference.cpp
)

target_link_libraries(
  gtsam_reference
  PRIVATE
  gtsam
)

add_executable(
    gtsam_cost_eval
    gtsam_cost_eval.cpp
)

target_link_libraries(
    gtsam_cost_eval
    PRIVATE
    gtsam
)
