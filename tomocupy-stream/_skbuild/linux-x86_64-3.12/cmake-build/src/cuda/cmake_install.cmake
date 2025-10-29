# Install script for directory: /home/cxhpc/data/wt/tomocupy-stream/src/cuda

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-install/src")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_fourierrec.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_fourierrec.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrec.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_lprec.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_lprec.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprec.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_linerec.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_linerec.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerec.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_filter.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_filter.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filter.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_fourierrecfp16.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_fourierrecfp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_fourierrecfp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_lprecfp16.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_lprecfp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_lprecfp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_linerecfp16.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_linerecfp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_linerecfp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE FILE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/cfunc_filterfp16.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tomocupy_stream" TYPE MODULE FILES "/home/cxhpc/data/wt/tomocupy-stream/_skbuild/linux-x86_64-3.12/cmake-build/src/cuda/_cfunc_filterfp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tomocupy_stream/_cfunc_filterfp16.so")
    endif()
  endif()
endif()

