# Unit tests
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
# TopOptObjFun
add_executable(TopOptObjFunTest test_topoptobjfun.cpp)
target_link_libraries(TopOptObjFunTest PRIVATE ${REQ_LIBS} Catch2::Catch2WithMain)
add_test(TopOptObjFunTest TopOptObjFunTest)
