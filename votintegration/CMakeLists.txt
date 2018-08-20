PROJECT(static)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Try to find TraX header and library ...
FIND_PACKAGE(trax REQUIRED COMPONENTS core)
LINK_DIRECTORIES(${TRAX_LIBRARY_DIRS})
LINK_LIBRARIES(${TRAX_LIBRARIES})
INCLUDE_DIRECTORIES(AFTER ${TRAX_INCLUDE_DIRS})

ADD_EXECUTABLE(static_c static.c) # Generate executable for C tracker
ADD_EXECUTABLE(static_cpp static.cpp) # Generate executable for C++ tracker

FIND_PACKAGE(OpenCV)

IF (OpenCV_FOUND)
ADD_EXECUTABLE(ncc ncc.cpp) # Generate executable for C tracker
INCLUDE_DIRECTORIES(AFTER ${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(ncc ${OpenCV_LIBS}) # Link with trax library
ENDIF()
