add_library(dstatesai_common
    common/common.cc
    )
target_compile_features(dstatesai_common PUBLIC cxx_std_17)
target_include_directories(dstatesai_common 
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
    )

add_library(dstatesai_client client/client.cc)
target_link_libraries(dstatesai_client PRIVATE dstatesai_common)

add_library(dstatesai_server server/server.cc)
target_link_libraries(dstatesai_server PRIVATE dstatesai_common)

add_executable(dstatesai_launcher launcher/launcher.cc)
target_link_libraries(dstatesai_launcher PRIVATE MPI::MPI_CXX dstatesai_server)


install(TARGETS dstatesai_common dstatesai_client dstatesai_server dstatesai_launcher EXPORT DStatesAITargets
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
