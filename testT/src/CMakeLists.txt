project(testT)

ADD_EXECUTABLE( main main.cpp)



# SET("PCL_DIR" "/home/jin/Third_party_packages/pcl-pcl-1.9.1/build")
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )


SET("OpenCV_DIR" "/home/jin/Third_party_packages/opencv-2.4.13.6/build")
find_package( OpenCV 2.4 REQUIRED )

ADD_LIBRARY( SLAMbase SLAMbase.cpp )
TARGET_LINK_LIBRARIES( SLAMbase
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES} )

ADD_EXECUTABLE( testT testT.cpp )
TARGET_LINK_LIBRARIES( testT
   SLAMbase
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES} )
