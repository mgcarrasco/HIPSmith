cmake_minimum_required(VERSION 3.15)

project(HIPSmith)

include(CheckIncludeFile)

set(HIPSmith_PACKAGE "HIPSmith")
set(HIPSmith_PACKAGE_BUGREPORT "mail@example.org")
set(HIPSmith_PACKAGE_NAME "HIPSmith")
set(HIPSmith_PACKAGE_STRING "HIPSmith x.y.z")
set(HIPSmith_PACKAGE_TARNAME "HIPSmith")
set(HIPSmith_PACKAGE_VERSION "x.y.z")
set(HIPSmith_VERSION "x.y.z")

execute_process(COMMAND "git" "show" "-s" "--format=%h" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE HIPSmith_GIT_HASH)
string(STRIP "${HIPSmith_GIT_HASH}" HIPSmith_GIT_HASH)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-long-long -Wall -std=c++11")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_definitions(-DGIT_VERSION=\"${HIPSmith_GIT_HASH}\")
add_definitions(-DTARGET_CPU_${CMAKE_SYSTEM_PROCESSOR}=1)

add_definitions(-DPACKAGE_NAME=\"${HIPSmith_PACKAGE_NAME}\")
add_definitions(-DPACKAGE_TARNAME=\"${HIPSmith_PACKAGE_TARNAME}\")
add_definitions(-DPACKAGE_VERSION=\"${HIPSmith_PACKAGE_VERSION}\")
add_definitions(-DPACKAGE_STRING=\"${HIPSmith_PACKAGE_STRING}\")
add_definitions(-DPACKAGE_BUGREPORT=\"${HIPSmith_PACKAGE_BUGREPORT}\")
add_definitions(-DPACKAGE_URL=\"${HIPSmith_PACKAGE_URL}\")
add_definitions(-DPACKAGE=\"${HIPSmith_PACKAGE}\")
add_definitions(-DVERSION=\"${HIPSmith_VERSION}\")

check_include_file("dlfcn.h" HAVE_DLFCN_H)
check_include_file("inttypes.h" HAVE_INTTYPES_H)
check_include_file("memory.h" HAVE_MEMORY_H)
check_include_file("stdint.h" HAVE_STDINT_H)
check_include_file("stdlib.h" HAVE_STDLIB_H)
check_include_file("strings.h" HAVE_STRINGS_H)
check_include_file("string.h" HAVE_STRING_H)
check_include_file("sys/stat.h" HAVE_SYS_STAT_H)
check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
check_include_file("unistd.h" HAVE_UNISTD_H)

if(WIN32)
    set(HIPSmith_WINDOWS_SOURCES
        runtime/windows/rand48.h
        runtime/windows/_rand48.c
        runtime/windows/lrand48.c
        runtime/windows/srand48.c
    )
endif()

include_directories(src)

add_executable(HIPSmith
    #CSmith files
    ${HIPSmith_WINDOWS_SOURCES}
    src/AbsExtension.cpp
    src/AbsExtension.h
    src/AbsProgramGenerator.cpp
    src/AbsProgramGenerator.h
    src/AbsRndNumGenerator.cpp
    src/AbsRndNumGenerator.h
    src/ArrayVariable.cpp
    src/ArrayVariable.h
    src/Attribute.cpp     
    src/Attribute.h
    src/Block.cpp
    src/Block.h
    src/Bookkeeper.cpp
    src/Bookkeeper.h
    src/CFGEdge.cpp
    src/CFGEdge.h
    src/CGContext.cpp
    src/CGContext.h
    src/CGOptions.cpp
    src/CGOptions.h
    src/CVQualifiers.cpp
    src/CVQualifiers.h
    src/Common.h
    src/CommonMacros.h
    src/CompatibleChecker.cpp
    src/CompatibleChecker.h
    src/Constant.cpp
    src/Constant.h
    src/CoverageTestExtension.cpp
    src/CoverageTestExtension.h
    src/CrestExtension.cpp
    src/CrestExtension.h
    src/DFSOutputMgr.cpp
    src/DFSOutputMgr.h
    src/DFSProgramGenerator.cpp
    src/DFSProgramGenerator.h
    src/DFSRndNumGenerator.cpp
    src/DFSRndNumGenerator.h
    src/DefaultOutputMgr.cpp
    src/DefaultOutputMgr.h
    src/DefaultProgramGenerator.cpp
    src/DefaultProgramGenerator.h
    src/DefaultRndNumGenerator.cpp
    src/DefaultRndNumGenerator.h
    src/DepthSpec.cpp
    src/DepthSpec.h
    src/Effect.cpp
    src/Effect.h
    src/Enumerator.h
    src/Error.cpp
    src/Error.h
    src/Expression.cpp
    src/Expression.h
    src/ExpressionAssign.cpp
    src/ExpressionAssign.h
    src/ExpressionComma.cpp
    src/ExpressionComma.h
    src/ExpressionFuncall.cpp
    src/ExpressionFuncall.h
    src/ExpressionVariable.cpp
    src/ExpressionVariable.h
    src/ExtensionMgr.cpp
    src/ExtensionMgr.h
    src/ExtensionValue.cpp
    src/ExtensionValue.h
    src/Fact.cpp
    src/Fact.h
    src/FactMgr.cpp
    src/FactMgr.h
    src/FactPointTo.cpp
    src/FactPointTo.h
    src/FactUnion.cpp
    src/FactUnion.h
    src/Filter.cpp
    src/Filter.h
    src/Finalization.cpp
    src/Finalization.h
    src/Function.cpp
    src/Function.h
    src/FunctionInvocation.cpp
    src/FunctionInvocation.h
    src/FunctionInvocationBinary.cpp
    src/FunctionInvocationBinary.h
    src/FunctionInvocationUnary.cpp
    src/FunctionInvocationUnary.h
    src/FunctionInvocationUser.cpp
    src/FunctionInvocationUser.h
    src/KleeExtension.cpp
    src/KleeExtension.h
    src/Lhs.cpp
    src/Lhs.h
    src/LinearSequence.cpp
    src/LinearSequence.h
    src/OutputMgr.cpp
    src/OutputMgr.h
    src/PartialExpander.cpp
    src/PartialExpander.h
    src/Probabilities.cpp
    src/Probabilities.h
    src/ProbabilityTable.h
    src/RandomNumber.cpp
    src/RandomNumber.h
    #src/RandomProgramGenerator.cpp
    src/SafeOpFlags.cpp
    src/SafeOpFlags.h
    src/Sequence.cpp
    src/Sequence.h
    src/SequenceFactory.cpp
    src/SequenceFactory.h
    src/SequenceLineParser.h
    src/Statement.cpp
    src/Statement.h
    src/StatementArrayOp.cpp
    src/StatementArrayOp.h
    src/StatementAssign.cpp
    src/StatementAssign.h
    src/StatementBreak.cpp
    src/StatementBreak.h
    src/StatementContinue.cpp
    src/StatementContinue.h
    src/StatementExpr.cpp
    src/StatementExpr.h
    src/StatementFor.cpp
    src/StatementFor.h
    src/StatementGoto.cpp
    src/StatementGoto.h
    src/StatementIf.cpp
    src/StatementIf.h
    src/StatementReturn.cpp
    src/StatementReturn.h
    src/StringUtils.cpp
    src/StringUtils.h
    src/Type.cpp
    src/Type.h
    src/Variable.cpp
    src/Variable.h
    src/VariableSelector.cpp
    src/VariableSelector.h
    src/VectorFilter.cpp
    src/VectorFilter.h
    src/platform.cpp
    src/platform.h
    src/random.cpp
    src/random.h
    src/util.cpp
    src/util.h
    #HIPSmith files
    src/HIPSmith/HIPOutputMgr.cpp
    src/HIPSmith/HIPOutputMgr.h
    src/HIPSmith/HIPOptions.h
    src/HIPSmith/HIPOptions.cpp
    src/HIPSmith/HIPProgramGenerator.cpp
    src/HIPSmith/HIPProgramGenerator.h
    src/HIPSmith/Globals.cpp
    src/HIPSmith/Globals.h
    src/HIPSmith/HIPRandomProgramGenerator.cpp
    src/HIPSmith/ExpressionVector.cpp
    src/HIPSmith/HIPExpression.cpp
    src/HIPSmith/HIPVariable.cpp
    src/HIPSmith/Vector.cpp
    src/HIPSmith/HIPStatement.cpp
    src/HIPSmith/StatementHIPSync.cpp
    src/HIPSmith/FunctionInvocationBuiltIn.cpp
)

find_program(M4_EXECUTABLE m4 DOC "The M4 macro processor")

if(M4_EXECUTABLE)
    set(SAFE_MATH_HEADERS
        ${CMAKE_BINARY_DIR}/safe_math_macros.h
    )

    add_custom_target(SAFE_MATH_HEADERS ALL
        DEPENDS ${SAFE_MATH_HEADERS}
    )

    add_custom_command(
        OUTPUT safe_math_macros.h
        COMMAND ${M4_EXECUTABLE} ${CMAKE_SOURCE_DIR}/runtime/safe_math_macros.m4 > safe_math_macros.h
        DEPENDS ${CMAKE_SOURCE_DIR}/runtime/safe_math_macros.m4
        VERBATIM
    )

    install(TARGETS HIPSmith
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )

    install(FILES ${SAFE_MATH_HEADERS}
        DESTINATION include/HIPSmith
    )
else()
    message(WARNING "Cannot build the safe math runtime header files because m4 was not found")
endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/runtime/HIPSmith.h"
    "${CMAKE_BINARY_DIR}/HIPSmith.h"
    COPYONLY
)

configure_file(
    "${CMAKE_SOURCE_DIR}/runtime/csmith.h"
    "${CMAKE_BINARY_DIR}/csmith.h"
    COPYONLY
)

configure_file(
    "${CMAKE_SOURCE_DIR}/scripts/test_hipsmith.py"
    "${CMAKE_BINARY_DIR}/test_hipsmith.py"
    COPYONLY
)
