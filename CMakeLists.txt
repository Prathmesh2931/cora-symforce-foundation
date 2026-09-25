cmake_minimum_required(VERSION 3.16)

project(
    cora_symforce_foundation
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(
    cora_symforce
    src/pgo_cost.cpp
    src/sesync_matrices.cpp
    src/simplified_pgo.cpp
    src/toy_problem.cpp
)

target_include_directories(
    cora_symforce
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    cora_symforce
    PUBLIC
        Eigen3::Eigen
)

target_compile_options(
    cora_symforce
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(
    reference_pgo
    apps/reference_pgo.cpp
)

target_link_libraries(
    reference_pgo
    PRIVATE
        cora_symforce
)

add_executable(
    matrix_equivalence
    apps/matrix_equivalence.cpp
)

target_link_libraries(
    matrix_equivalence
    PRIVATE
        cora_symforce
)

include(CTest)

if(BUILD_TESTING)
    add_executable(
        test_reference_pgo
        tests/test_reference_pgo.cpp
    )

    target_link_libraries(
        test_reference_pgo
        PRIVATE
            cora_symforce
    )

    add_test(
        NAME reference_pgo_tests
        COMMAND test_reference_pgo
    )

    add_executable(
        test_matrix_equivalence
        tests/test_matrix_equivalence.cpp
    )

    target_link_libraries(
        test_matrix_equivalence
        PRIVATE
            cora_symforce
    )

    add_test(
        NAME matrix_equivalence_tests
        COMMAND test_matrix_equivalence
    )
endif()

add_executable(
    simplified_equivalence
    apps/simplified_equivalence.cpp
)

target_link_libraries(
    simplified_equivalence
    PRIVATE
        cora_symforce
)

if(BUILD_TESTING)
    add_executable(
        test_simplified_equivalence
        tests/test_simplified_equivalence.cpp
    )

    target_link_libraries(
        test_simplified_equivalence
        PRIVATE
            cora_symforce
    )

    add_test(
        NAME simplified_equivalence_tests
        COMMAND test_simplified_equivalence
    )
endif()
