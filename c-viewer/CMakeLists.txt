cmake_minimum_required(VERSION 3.20)
project(c_image_viewer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Options
option(WITH_DICOM "Enable DICOM (.dcm) support via GDCM" ON)
option(WITH_OPENSLIDE "Enable SVS support via OpenSlide" ON)

# Dependencies
# SDL2 for window and input (support both CONFIG and Module modes)
set(SDL2_FOUND_CONFIG FALSE)
find_package(SDL2 CONFIG QUIET)
if (TARGET SDL2::SDL2)
  set(SDL2_FOUND_CONFIG TRUE)
else()
  find_package(SDL2 QUIET)
endif()
if (NOT SDL2_FOUND_CONFIG AND NOT SDL2_FOUND AND NOT TARGET SDL2::SDL2)
  message(FATAL_ERROR "SDL2 not found. Install via vcpkg (recommended) or set SDL2_DIR/CMAKE_PREFIX_PATH to the SDL2Config.cmake folder.")
endif()

# OpenSlide (optional)
if (WITH_OPENSLIDE)
  find_path(OPENSLIDE_INCLUDE_DIR openslide/openslide.h)
  find_library(OPENSLIDE_LIBRARY NAMES openslide)
  if (OPENSLIDE_INCLUDE_DIR AND OPENSLIDE_LIBRARY)
    message(STATUS "OpenSlide found: ${OPENSLIDE_LIBRARY}")
    add_definitions(-DHAVE_OPENSLIDE)
  else()
    message(WARNING "OpenSlide not found; SVS disabled")
    set(WITH_OPENSLIDE OFF)
  endif()
endif()

# GDCM (optional) for DICOM
if (WITH_DICOM)
  find_package(GDCM QUIET)
  if (GDCM_FOUND)
    message(STATUS "GDCM found")
    add_definitions(-DHAVE_GDCM)
  else()
    message(WARNING "GDCM not found; basic DICOM disabled")
    set(WITH_DICOM OFF)
  endif()
endif()

# stb_image (vendored single header)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_executable(c_image_viewer
  src/main.c
  src/viewer.c
  src/loader_common.c
  src/loader_raster.c
  src/loader_svs.c
  src/loader_dcm.c
)

target_include_directories(c_image_viewer PRIVATE src)

# Link SDL2 depending on how it was found
if (TARGET SDL2::SDL2)
  target_link_libraries(c_image_viewer PRIVATE SDL2::SDL2 SDL2::SDL2main)
elseif (SDL2_FOUND)
  # Legacy FindSDL2.cmake exposes SDL2_INCLUDE_DIRS and SDL2_LIBRARIES
  target_include_directories(c_image_viewer PRIVATE ${SDL2_INCLUDE_DIRS})
  target_link_libraries(c_image_viewer PRIVATE ${SDL2_LIBRARIES})
else()
  message(FATAL_ERROR "SDL2 found state is inconsistent. Please ensure SDL2 is installed and discoverable by CMake.")
endif()

# Install rules (for CPack / MSI)
install(TARGETS c_image_viewer RUNTIME DESTINATION bin)

# CPack configuration for MSI via WiX
set(CPACK_PACKAGE_NAME "C Image Viewer")
set(CPACK_PACKAGE_VENDOR "Pathothai")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CImageViewer")
if (WIN32)
  set(CPACK_GENERATOR "WIX")
  # Stable GUID for upgrades across versions (generate once and keep)
  set(CPACK_WIX_UPGRADE_GUID "A2B5C0B7-5E35-4F3B-8A0E-6C9E0CEB4D51")
  # Optional: create Start Menu shortcut
  set(CPACK_CREATE_DESKTOP_LINKS c_image_viewer)
endif()

include(CPack)

if (WITH_OPENSLIDE)
  target_include_directories(c_image_viewer PRIVATE ${OPENSLIDE_INCLUDE_DIR})
  target_link_libraries(c_image_viewer PRIVATE ${OPENSLIDE_LIBRARY})
endif()

if (WITH_DICOM AND GDCM_FOUND)
  target_link_libraries(c_image_viewer PRIVATE gdcmMSFF)
endif()

# Windows resources (icon/version) could be added here

