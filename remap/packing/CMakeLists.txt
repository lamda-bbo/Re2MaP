add_subdirectory(common)
add_subdirectory(representation)
add_subdirectory(solver)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION remap/packing
    )
