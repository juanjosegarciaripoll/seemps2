#pragma once
#include "core.h"

namespace seemps {

using Weight = py::object;

py::object _begin_environment(int D = 1);
py::object _update_left_environment(py::object A, py::object B, py::object rho);
py::object _update_right_environment(py::object A, py::object B,
                                     py::object rho);
Weight _end_environment(py::object rho);
Weight _join_environments(py::object rhoL, py::object rhoR);

Weight scprod(py::object A, py::object B);

} // namespace seemps
