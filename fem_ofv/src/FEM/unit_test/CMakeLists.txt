enable_testing()
# Unit tests
# LinearTriangle
add_executable(LinTriTest test_lintri.cpp)
target_link_libraries(LinTriTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(LinTriTest LinTriTest)
# LinearQuadrilateral
add_executable(LinQuadTest test_linquad.cpp)
target_link_libraries(LinQuadTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(LinQuadTest LinQuadTest)
# LinTetra
add_executable(LinTetraTest test_lintetra.cpp)
target_link_libraries(LinTetraTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(LinTetraTest LinTetraTest)
# LinHex
add_executable(LinHexTest test_linhex.cpp)
target_link_libraries(LinHexTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(LinHexTest LinHexTest)
# Element, tests connectivity functions
add_executable(ElementTest test_element.cpp)
target_link_libraries(ElementTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(ElementTest ElementTest)
# ElemEdge, tests connectivity functions
add_executable(ElemEdgeTest test_elemedge.cpp)
target_link_libraries(ElemEdgeTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(ElemEdgeTest ElemEdgeTest)
# Mesh2D
add_executable(Mesh2DTest test_mesh2d.cpp)
target_link_libraries(Mesh2DTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(Mesh2DTest Mesh2DTest)
# Mesh3D
add_executable(Mesh3DTest test_mesh3d.cpp)
target_link_libraries(Mesh3DTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(Mesh3DTest Mesh3DTest)
# BoundaryCondition
add_executable(BoundaryConditionTest test_boundarycondition.cpp)
target_link_libraries(BoundaryConditionTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(BoundaryConditionTest BoundaryConditionTest)
# LoadCondition
add_executable(LoadConditionTest test_loadcondition.cpp)
target_link_libraries(LoadConditionTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(LoadConditionTest LoadConditionTest)
# FEMProblem
add_executable(FEMProblemTest test_femproblem.cpp)
target_link_libraries(FEMProblemTest PRIVATE Catch2::Catch2WithMain ${FEM_LIB} ${TOPOLOGIESCORE_LIB} )
add_test(FEMProblemTest FEMProblemTest)
# copy input files
configure_file(./inputfiles/bctesttrimesh.txt ${CMAKE_CURRENT_BINARY_DIR}/bctesttrimesh.txt COPYONLY)
configure_file(./inputfiles/testtri.msh ${CMAKE_CURRENT_BINARY_DIR}/testtri.msh COPYONLY)
configure_file(./inputfiles/hexelemmat.txt ${CMAKE_CURRENT_BINARY_DIR}/hexelemmat.txt COPYONLY)
