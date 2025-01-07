# FindCython.cmake

if(NOT CYTHON_EXECUTABLE)
    # 尋找 cython 執行檔
    find_program(CYTHON_EXECUTABLE NAMES cython cython.py cython3)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cython
                                REQUIRED_VARS CYTHON_EXECUTABLE)

mark_as_advanced(CYTHON_EXECUTABLE)

if(CYTHON_FOUND AND NOT TARGET Cython::Cython)
    add_executable(Cython::Cython IMPORTED)
    set_property(TARGET Cython::Cython PROPERTY
                IMPORTED_LOCATION ${CYTHON_EXECUTABLE})
endif()

# 定義添加 Cython 目標的巨集
macro(add_cython_target _name _sourcefile)
    add_custom_command(
        OUTPUT ${_name}.cpp
        COMMAND ${CYTHON_EXECUTABLE}
        ARGS --cplus -3 --embed ${_sourcefile} -o ${_name}.cpp
        DEPENDS ${_sourcefile}
        COMMENT "Generating C++ source from Cython file ${_sourcefile}"
    )
endmacro()
