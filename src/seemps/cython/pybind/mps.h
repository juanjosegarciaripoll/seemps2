#pragma once
#include "core.h"
#include "strategy.h"

namespace seemps
{

using Weight = py::object;

py::object _begin_environment(int D = 1);
py::object _update_left_environment(py::object A, py::object B, py::object rho);
py::object _update_right_environment(py::object A, py::object B,
                                     py::object rho);
Weight _end_environment(py::object rho);
Weight _join_environments(py::object rhoL, py::object rhoR);

Weight scprod(py::object A, py::object B);

py::object schmidt_weights(py::object A);

std::tuple<int, double>
_update_in_canonical_form_right(py::list state, py::object A, int site,
                                const Strategy& truncation);
std::tuple<int, double>
_update_in_canonical_form_left(py::list state, py::object A, int site,
                               const Strategy& truncation);
double _canonicalize(py::list state, int center, const Strategy& truncation);
double _recanonicalize(py::list state, int oldcenter, int newcenter,
                       const Strategy& truncation);
std::tuple<py::object, py::object, double>
_left_orth_2site(py::object AA, const Strategy& strategy);
std::tuple<py::object, py::object, double>
_right_orth_2site(py::object AA, const Strategy& strategy);
double _update_canonical_2site_left(py::list state, py::object A, int site,
                                    const Strategy& strategy);
double _update_canonical_2site_right(py::list state, py::object A, int site,
                                     const Strategy& strategy);
} // namespace seemps
