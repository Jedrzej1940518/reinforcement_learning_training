#include <pybind11/pybind11.h>

#include "ShootingGame.hpp"  // Your game's header file

namespace py = pybind11;

// Define the module named "shooting_game"
PYBIND11_MODULE(shooting_game, m) {
    py::class_<ShootingGame>(m, "ShootingGame")
        .def(py::init<>())  // Binding the constructor
        .def("step", &ShootingGame::step)  // Binding the step method
        .def("reset", &ShootingGame::reset)  // Binding the reset method
        .def("init_render", &ShootingGame::init_render)
        .def("draw", &ShootingGame::draw);

    // Add other bindings here
}

