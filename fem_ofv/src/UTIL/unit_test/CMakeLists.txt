enable_testing()
# Unit tests
# Point2D
add_executable(Point2dTest test_point2d.cpp)
target_link_libraries(Point2dTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} NLopt::nlopt ${TOPOLOGIESCORE_LIB})
add_test(Point2dTest Point2dTest)
# Point3D
add_executable(Point3dTest test_point3d.cpp)
target_link_libraries(Point3dTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} NLopt::nlopt ${TOPOLOGIESCORE_LIB})
add_test(Point3dTest Point3dTest)
# GeometricEntity
add_executable(GeometricEntityTest test_geometricentity.cpp)
target_link_libraries(GeometricEntityTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} NLopt::nlopt ${TOPOLOGIESCORE_LIB})
add_test(GeometricEntityTest GeometricEntityTest)
# CoordinateSystem
add_executable(CoordinateSystemTest test_coordinatesystem.cpp)
target_link_libraries(CoordinateSystemTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} NLopt::nlopt ${TOPOLOGIESCORE_LIB})
add_test(CoordinateSystemTest CoordinateSystemTest)

