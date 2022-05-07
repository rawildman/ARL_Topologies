# Created by the script cgal_create_cmake_script_noqt
# This is the CMake script for compiling a CGAL application.


project( optfun_sc )

cmake_minimum_required(VERSION 2.6.2)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 2.6)
  if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" VERSION_GREATER 2.8.3)
    cmake_policy(VERSION 2.8.4)
  else()
    cmake_policy(VERSION 2.6)
  endif()
endif()

find_package(pugixml)

if ( CGAL_FOUND )
# Definitions
	if(DEFINE_USE_MPI)
		add_definitions(-DUSE_MPI)
	endif(DEFINE_USE_MPI)
	if(NOT DEFINED TOPOLOGIES_ROOT)
		set(TOPOLOGIES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../topologies/src)
		set(TOPOLOGIESCORE_LIB ${CMAKE_CURRENT_BINARY_DIR}/../../../topologies/src/libtopologiescore.so)
	else(NOT DEFINED TOPOLOGIES_ROOT)
		# Note that if TOPOLOGIES_ROOT is defined, it is assumed that libtopologiescore.so is in that directory:
		# This is so that the fem_ofv compilation can be separated from topologies, if desired
		set(TOPOLOGIES_SRC ${TOPOLOGIES_ROOT}/topologies/src)
		set(TOPOLOGIESCORE_LIB ${TOPOLOGIES_ROOT}/lib/libtopologiescore.so)
	endif(NOT DEFINED TOPOLOGIES_ROOT)
# Set include directories
	include( ${CGAL_USE_FILE} )
	include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
	include_directories (BEFORE ${TOPOLOGIES_SRC})
	include_directories (BEFORE "../UTIL/")
	include_directories (BEFORE "../FEM/")
	#include_directories (BEFORE ${PUGIXML_INC_DIR})
# Set source files
  file (GLOB ALLSOURCE "*.cpp")
	set (FULL_SOURCE ${ALLSOURCE} )
# Libraries
# Set executable
	add_library (optfun_sc SHARED ${FULL_SOURCE})
	set_property(TARGET optfun_sc PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
	target_link_libraries( optfun_sc pugixml::pugixml femofv )
# Install
	install (TARGETS optfun_sc LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
else()
    message(STATUS "NOTICE: This library requires the CGAL library, and will not be compiled.")
endif()

