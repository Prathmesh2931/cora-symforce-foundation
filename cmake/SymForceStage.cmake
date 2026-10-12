set(
    SYMFORCE_GENERATED_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)

set(
    SYMFORCE_EDGE_RESIDUAL_HEADER
    "${SYMFORCE_GENERATED_DIR}/edge_residual.h"
)

set(
    SYMFORCE_EDGE_JACOBIAN_HEADER
    "${SYMFORCE_GENERATED_DIR}/edge_jacobian.h"
)


if(
    EXISTS "${SYMFORCE_EDGE_RESIDUAL_HEADER}"
    AND
    EXISTS "${SYMFORCE_EDGE_JACOBIAN_HEADER}"
)

    # Add the generated-kernel wrapper to the existing library.
    target_sources(
        cora_symforce
        PRIVATE
            src/symforce_pgo_cost.cpp
    )

    target_include_directories(
        cora_symforce
        PRIVATE
            "${SYMFORCE_GENERATED_DIR}"
    )


    # Low-level generated residual/Jacobian equivalence test.
    add_executable(
        test_symforce_edge_equivalence
        tests/test_symforce_edge_equivalence.cpp
    )

    target_include_directories(
        test_symforce_edge_equivalence
        PRIVATE
            "${SYMFORCE_GENERATED_DIR}"
    )

    target_link_libraries(
        test_symforce_edge_equivalence
        PRIVATE
            Eigen3::Eigen
    )


    # Graph-level reference-vs-SymForce PGO test.
    add_executable(
        test_symforce_pgo_cost
        tests/test_symforce_pgo_cost.cpp
    )

    target_link_libraries(
        test_symforce_pgo_cost
        PRIVATE
            cora_symforce
            Eigen3::Eigen
    )


    if(BUILD_TESTING)

        add_test(
            NAME symforce_edge_equivalence
            COMMAND test_symforce_edge_equivalence
        )

        add_test(
            NAME symforce_pgo_cost
            COMMAND test_symforce_pgo_cost
        )

    endif()


else()

    message(
        STATUS
        "SymForce generated headers not found. "
        "Run codegen/python/generate_pgo_residuals.py first."
    )

endif()
