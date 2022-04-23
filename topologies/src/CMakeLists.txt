# Created by the script cgal_create_cmake_script_noqt
# This is the CMake script for compiling a CGAL application.


project( topologies )

cmake_minimum_required(VERSION 2.6.2)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 2.6)
  if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" VERSION_GREATER 2.8.3)
    cmake_policy(VERSION 2.8.4)
  else()
    cmake_policy(VERSION 2.6)
  endif()
endif()

add_definitions(-DCGAL_MESH_3_NO_DEPRECATED_SURFACE_INDEX
                -DCGAL_MESH_3_NO_DEPRECATED_C3T3_ITERATORS)
find_package(CGAL QUIET COMPONENTS Core )
find_package(NLopt)
find_package(pugixml)

if ( CGAL_FOUND )
# Compiler flags
  if(DEFINE_USE_MPI)
	  add_definitions(-DUSE_MPI)
  endif(DEFINE_USE_MPI)
# Set include directories
	include( ${CGAL_USE_FILE} )
	include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
	include_directories (BEFORE "IO/")
	include_directories (BEFORE "OBJFUN/")
	include_directories (BEFORE "OPT/")
	include_directories (BEFORE "REP/")
	include_directories (BEFORE "UTIL/")
# Set source files
	add_subdirectory (IO)
	add_subdirectory (OBJFUN)
	add_subdirectory (OPT)
	add_subdirectory (REP)
	add_subdirectory (UTIL)
# Required libraries
	set (DL_LIB libdl.so)
# RPATH for install
	set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# Set library
	# TO library
	add_library(topologiescore SHARED ${TOLIB_SOURCE})
	target_link_libraries(topologiescore PRIVATE pugixml::pugixml ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES} ${DL_LIB})
	set_property(TARGET topologiescore PROPERTY CXX_STANDARD 11)
	set_property(TARGET topologiescore PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET topologiescore PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
	set(topologies_LIB topologiescore)
	# Opt library
	add_library(topologiesopt SHARED ${OPTLIB_SOURCE})
	target_link_libraries(topologiesopt PRIVATE topologiescore NLopt::nlopt ${DL_LIB})
	set_property(TARGET topologiesopt PROPERTY CXX_STANDARD 11)
	set_property(TARGET topologiesopt PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET topologiesopt PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
	set(topologiesopt_LIB topologiesopt)

# Set executable
	add_subdirectory (MAIN)
	add_executable(topologies ${MAIN_SOURCE})
	set_property(TARGET topologies PROPERTY CXX_STANDARD 11)
	set_property(TARGET topologies PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET topologies PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
	add_to_cached_list( CGAL_EXECUTABLE_TARGETS topologies )
# Link the executable to CGAL and third-party libraries
	set (REQ_LIBS ${topologies_LIB} ${topologiesopt_LIB} pugixml::pugixml)
	target_link_libraries(topologies ${REQ_LIBS} ${DL_LIB})
# Set up unit and regression testing
if(COMPILE_TESTS)
	enable_testing()
	find_package(Catch2 3 REQUIRED)
	add_subdirectory (IO/unit_test)
	add_subdirectory (OBJFUN/unit_test)
	add_subdirectory (OPT/unit_test)
	add_subdirectory (REP/unit_test)
	add_subdirectory (UTIL/unit_test)
	# Note that the following tests may take a long time to complete:
	add_subdirectory (TEST)
endif(COMPILE_TESTS)
# Install
	install (TARGETS topologiescore LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
	install (TARGETS topologiesopt LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
	install (TARGETS topologies RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
# Install example files
	add_subdirectory (EX_INPUT)
else()
    message(STATUS "NOTICE: The CGAL library is required and will not be compiled.")
endif()

