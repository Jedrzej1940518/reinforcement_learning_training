cmake_minimum_required(VERSION 3.4)
project(ShootingGameModule)

# Execute "which python" to find the path to the Python interpreter
execute_process(
    COMMAND which python
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the Python interpreter found by which
set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})

# Find Python3 package, ensuring it picks up the Python version from which command
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

pybind11_add_module(shooting_game ShootingGameModule.cpp) # or wherever your binding code is

target_link_libraries(shooting_game PRIVATE pybind11::embed sfml-graphics sfml-window sfml-system)

set_target_properties(shooting_game PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)