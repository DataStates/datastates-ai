set(COMMON_LIBRARIES thallium)

add_library(evostore_client client/client.cpp)
target_link_libraries(evostore_client PUBLIC CUDA::cudart ${COMMON_LIBRARIES})

nanobind_add_module(dstates client/client-py-module.cpp client/client-py-impl.cpp)
target_link_libraries(dstates PRIVATE evostore_client)

add_library(evostore_server server/server.cpp)
target_link_libraries(evostore_server PRIVATE ${COMMON_LIBRARIES})

add_executable(evostore_slauncher server/simple_launcher.cpp)
target_link_libraries(evostore_slauncher PRIVATE evostore_server ${COMMON_LIBRARIES})

install(TARGETS evostore_client evostore_server dstates evostore_slauncher EXPORT DStatesAITargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT DStatesAITargets NAMESPACE DStatesAI:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DStatesAI/)

export(EXPORT DStatesAITargets FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/DStatesAITargets.cmake NAMESPACE DStatesAI::)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/DStatesAIConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DStatesAIConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DStatesAI
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DStatesAIConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DStatesAIConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DStatesAIConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DStatesAI
)
