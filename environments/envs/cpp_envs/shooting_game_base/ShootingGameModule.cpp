#include <pybind11/pybind11.h>

#include "ShootingGame.hpp"  // Your game's header file

namespace py = pybind11;

// Define the module named "shooting_game_module"
PYBIND11_MODULE(shooting_game_module, m) {
    py::class_<ShootingGame>(m, "ShootingGame")
        .def(py::init<>())  // Binding the constructor
        .def("step", &ShootingGame::step)  // Binding the step method
        .def("reset", &ShootingGame::reset)  // Binding the reset method
        .def("init_human_render", &ShootingGame::init_human_render)
        .def("init_rgb_render", &ShootingGame::init_rgb_render)
        .def("render_rbg", &ShootingGame::render_rbg)
        .def("render_human", &ShootingGame::render_human);
    // Add other bindings here
}

