# ---------------------------------------------------------------
# Programmer(s): Radu Serban, David J. Gardner, Cody J. Balos,
#                and Slaven Peles @ LLNL
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2023, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------
# Top level CMakeLists.txt for SUNDIALS (for cmake build system)
# ---------------------------------------------------------------

# ===============================================================
# Initial setup.
# ===============================================================

cmake_minimum_required(VERSION 3.12)

# Project SUNDIALS (initially only C supported)
# sets PROJECT_SOURCE_DIR and PROJECT_BINARY_DIR variables.
project(SUNDIALS C)

# Specify the location of additional CMAKE modules
set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  ${PROJECT_SOURCE_DIR}/cmake/macros
  ${PROJECT_SOURCE_DIR}/cmake/tpl
  )

# MACRO definitions
include(SundialsCMakeMacros)
include(CMakePrintHelpers)
include(CheckCSourceCompiles)
include(FindPackageHandleStandardArgs)


# Set some variables with info on the SUNDIALS project
set(PACKAGE_BUGREPORT "sundials-users@llnl.gov")
set(PACKAGE_NAME "SUNDIALS")
set(PACKAGE_STRING "SUNDIALS 6.5.1")
set(PACKAGE_TARNAME "sundials")

# Set SUNDIALS version numbers
sundials_git_version() # sets SUNDIALS_GIT_VERSION
message(STATUS "SUNDIALS_GIT_VERSION: ${SUNDIALS_GIT_VERSION}")

# (use "" for the version label if none is needed)
set(PACKAGE_VERSION_MAJOR "6")
set(PACKAGE_VERSION_MINOR "5")
set(PACKAGE_VERSION_PATCH "1")
set(PACKAGE_VERSION_LABEL "")

if(PACKAGE_VERSION_LABEL)
  set(PACKAGE_VERSION
      "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}-${PACKAGE_VERSION_LABEL}"
  )
else()
  set(PACKAGE_VERSION
      "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}"
  )
endif()

# Specify the VERSION and SOVERSION for shared libraries

set(arkodelib_VERSION "5.5.1")
set(arkodelib_SOVERSION "5")

set(cvodelib_VERSION "6.5.1")
set(cvodelib_SOVERSION "6")

set(cvodeslib_VERSION "6.5.1")
set(cvodeslib_SOVERSION "6")

set(idalib_VERSION "6.5.1")
set(idalib_SOVERSION "6")

set(idaslib_VERSION "5.5.1")
set(idaslib_SOVERSION "5")

set(kinsollib_VERSION "6.5.1")
set(kinsollib_SOVERSION "6")

set(cpodeslib_VERSION "0.0.0")
set(cpodeslib_SOVERSION "0")

set(nveclib_VERSION "6.5.1")
set(nveclib_SOVERSION "6")

set(sunmatrixlib_VERSION "4.5.1")
set(sunmatrixlib_SOVERSION "4")

set(sunlinsollib_VERSION "4.5.1")
set(sunlinsollib_SOVERSION "4")

set(sunnonlinsollib_VERSION "3.5.1")
set(sunnonlinsollib_SOVERSION "3")

set(sundialslib_VERSION
    "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}"
)
set(sundialslib_SOVERSION "${PACKAGE_VERSION_MAJOR}")

# ===============================================================
# Initial Setup
# ===============================================================

# Prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  print_error("In-source build prohibited.")
endif()

# Organize targets into folders when using an IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Get correct build paths automatically, but expose LIBDIR and
# INCLUDEDIR as a regular cache variable so that a user can more
# easily see what they were set to by GNUInstallDirs.
include(GNUInstallDirs)
mark_as_advanced(CLEAR CMAKE_INSTALL_LIBDIR)
mark_as_advanced(CLEAR CMAKE_INSTALL_INCLUDEDIR)

# Suffixes to use for static and shared targets.
set(_STATIC_LIB_SUFFIX
    "_static"
    CACHE INTERNAL "" FORCE
)
set(_SHARED_LIB_SUFFIX
    "_shared"
    CACHE INTERNAL "" FORCE
)

# A list of all the alias targets created.
set(_SUNDIALS_ALIAS_TARGETS ""
    CACHE INTERNAL "" FORCE)

# ===============================================================
# Build options to be processed first
# ===============================================================

include(SundialsDeprecated)
include(SundialsBuildOptionsPre)

# ===============================================================
# Options for external packages
# ===============================================================

include(SundialsTPLOptions)

# ===============================================================
# Options for examples
# ===============================================================

include(SundialsExampleOptions)

# ===============================================================
# Setup compilers
# ===============================================================

include(SundialsSetupCompilers)

# ===============================================================
# Setup third-party libraries
# ===============================================================

include(SundialsSetupTPLs)

# ===============================================================
# Build options to be proccessed last
# ===============================================================

include(SundialsBuildOptionsPost)

# ===============================================================
# At this point all the configuration options are set.
# Setup the sundials_config.h.
# ===============================================================

include(SundialsSetupConfig)

# ===============================================================
# Add src and tests; optionally add examples and unit tests.
# ===============================================================

# Add selected packages and modules to the build
add_subdirectory(src)

# Add selected examples to the build
if(_BUILD_EXAMPLES)
  include(SundialsSetupTesting)
  add_subdirectory(examples)
endif()

# Add benchmarks to the build
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Add unit tests to the build if they are enabled
if(SUNDIALS_TEST_UNITTESTS)
  add_subdirectory(test/unit_tests)
endif()

# ===============================================================
# Install configuration header files and license file.
# ===============================================================

# install sundials_export header file
install(FILES "${PROJECT_BINARY_DIR}/include/sundials/sundials_export.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sundials"
)

# install configured header file
install(FILES "${PROJECT_BINARY_DIR}/include/sundials/sundials_config.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sundials"
)

# install shared Fortran 2003 modules
if(BUILD_FORTRAN_MODULE_INTERFACE)
  # While the .mod files get generated for static and shared
  # libraries, they are identical. So only install one set
  # of the .mod files.
  if(BUILD_STATIC_LIBS)
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}_STATIC/
            DESTINATION ${Fortran_INSTALL_MODDIR}
    )
  else()
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}_SHARED/
            DESTINATION ${Fortran_INSTALL_MODDIR}
    )
  endif()
endif()

# install license and notice files
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sundials"
)
install(FILES "${PROJECT_SOURCE_DIR}/NOTICE"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sundials"
)

# create package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  SUNDIALSConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# install targets
install(
  EXPORT sundials-targets
  FILE SUNDIALSTargets.cmake
  NAMESPACE SUNDIALS::
  DESTINATION "${SUNDIALS_INSTALL_CMAKEDIR}"
)

# install SUNDIALSConfig.cmake
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/SUNDIALSConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/SUNDIALSConfig.cmake"
  INSTALL_DESTINATION "${SUNDIALS_INSTALL_CMAKEDIR}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SUNDIALSConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SUNDIALSConfigVersion.cmake"
        DESTINATION "${SUNDIALS_INSTALL_CMAKEDIR}"
)

# Export targets so build directory can be used directly
export(
  EXPORT sundials-targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/SUNDIALSTargets.cmake"
  NAMESPACE SUNDIALS::
)
