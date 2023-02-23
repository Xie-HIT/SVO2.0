cmake_minimum_required(VERSION 2.8.3)
project(dbow2_catkin)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(ExternalProject)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

ExternalProject_Add(dbow2_src
  # GIT_REPOSITORY git@github.com:dorian3d/DBoW2.git
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
  BUILD_COMMAND CXXFLAGS=-i${CATKIN_DEVEL_PREFIX}/include make
  INSTALL_COMMAND make install
)
ExternalProject_CatkinInstall(dbow2_src)

cs_add_library(${PROJECT_NAME} SHARED src/dependency_tracker.cc)
add_dependencies(${PROJECT_NAME} dbow2_src)
target_link_libraries(${PROJECT_NAME} 
  ${CATKIN_DEVEL_PREFIX}/lib/libDBoW2${CMAKE_SHARED_LIBRARY_SUFFIX})

cs_install()
cs_export(
  INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include
  CFG_EXTRAS dbow2-extras.cmake)
