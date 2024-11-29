#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>
#include <thread>

#include "helpers.h"

// Enable convenient time formats (12ms,..)
using namespace std::chrono_literals;

/// <summary>
/// Multiplies each image value by given factor.
/// Returns a copy of the image. The original image is unchanged.
/// </summary>
/// <param name="image">image to scale</param>
/// <param name="factor">multiplier</param>
/// <returns>copy of the image</returns>
ImageRGB scaleImageCopy(const ImageRGB& image, const float factor)
{
    auto result = ImageRGB(image.width, image.height);
    auto num_pixels = image.width * image.height;

    for (auto i = 0; i < num_pixels; i++) {
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    }

    return result;
}

/// <summary>
/// Multiplies each image value by given factor.
/// Modified the input image.
/// </summary>
/// <param name="image">image to scale</param>
/// <param name="factor">multiplier</param>
/// <param name="factor"></param>
void scaleImageInPlace(ImageRGB& image, const float factor)
{
    auto num_pixels = image.width * image.height;

    for (auto i = 0; i < num_pixels; i++) {
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    }
}

/// <summary>
/// A method that does a lot of small tasks.
/// </summary>
/// <param name="size">number of tasks</param>
void doLotOfWork(int size) 
{
        
    // Loop over tasks.
    #pragma omp parallel for
    for (auto i = 0; i < size; i++) 
    {
        // Simulate work by waiting 1ms.
        std::this_thread::sleep_for(1ms);
    }
}

/// <summary>
/// Finds a minimum value in the list.
/// </summary>
/// <param name="list"></param>
/// <returns></returns>
float getMinimumValue(const std::vector<float>& list)
{
    // Initialize the variable.
    float min_val = list[0];

    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    // Return minimum value.
    return min_val;
}


float getSum(const std::vector<float>& list)
{
    float sum = 0.0;
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    return sum;
}