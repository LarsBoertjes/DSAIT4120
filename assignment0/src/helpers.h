#pragma once
/*
This file contains useful function and definitions.
Do not ever edit this file - it will not be uploaded for evaluation.
If you want to modify any of the functions here (e.g. extend triangle test to quads),
copy the function "your_code_here.h" and give it a new name.
*/

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
DISABLE_WARNINGS_POP()

#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>

#ifdef _OPENMP
// Only if OpenMP is enabled.
#include <omp.h>
#endif

#include <framework/image.h>

/// <summary>
/// Aliases for Image classes.
/// </summary>
using ImageRGB = Image<glm::vec3>;

/// <summary>
/// Prints helpful information about OpenMP.
/// </summary>
void printOpenMPStatus() 
{
#ifdef _OPENMP
    // https://stackoverflow.com/questions/38281448/how-to-check-the-version-of-openmp-on-windows
    std::cout << "OpenMP version " << _OPENMP << " is ENABLED with " << omp_get_max_threads() << " threads." << std::endl;
#else
    std::cout << "OpenMP is DISABLED." << std::endl;
#endif
}