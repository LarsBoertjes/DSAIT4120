#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "helpers.h"


/*
 * Utility functions.
 */

/// <summary>
/// Bilinearly sample ImageFloat or ImageRGB using image coordinates [x,y].
/// Note that the coordinate start and end points align with the entire image and not just
/// the pixel centers (a half-pixel difference).
/// </summary>
/// <typeparam name="T">template type, can be ImageFloat or ImageRGB</typeparam>
/// <param name="image">input image</param>
/// <param name="pos">x,y position in px units</param>
/// <returns>interpolated pixel value (float or glm::vec3)</returns>
template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& pos_px)
{
    // Write a code that bilinearly interpolates values from a generic image (can contain either float or glm::vec3).
    // The pos_px input represents the (x,y) pixel coordinates of the sampled point where:
    //   [0, 0] = The left top corner of the left top (=first) pixel.
    //   [width, height] = The right bottom corner of the right bottom (=last) pixel.
    //   [0, height] = The left bottom corner of the left bottom pixel.
    // 
    // Take into account the size of individual pixel and the fact, that the "value" of pixel is conceptually stored in its center.
    //      => Example 1: For pos_px between centers of pixels, the method bilinearly interpolates between 4 nearest pixel.
    //      => Example 2: For pos_px corresponding to a center of a pixel, the method needs to return an exact value of that pixel. 
    //                    This is a natural property of any interpolation.
    // 
    // Therefore, steps are as follows:
    //     1. Determine the 4 nearest pixels.
    //     2. Bilinearly interpolate their values based on the position of the sampling point between them.
    // 
    // Note: The method is templated by parameter "T". This will be either float or glm::vec3 depending on whether the method
    // is called with ImageFloat or ImageRGB. Use either "T" or "auto" to define your variables and use glm::functions to handle both types.
    // Example:
    //    auto value = image.data[0] * 3; // both float and glm:vec3 support baisc operators
    //    T rounded_value = glm::round(image.data[0]); // glm::round will handle both glm::vec3 and float. 
    // Use glm API for further reference: https://glm.g-truc.net/0.9.9/api/a00241.html
    // 
    
    const int width = image.width;
    const int height = image.height;

    glm::vec2 clamped_pos = glm::clamp(pos_px, glm::vec2(0), glm::vec2(width, height));

    // Calculate the fractional and integer parts of the input position
    int x0 = static_cast<int>(std::floor(clamped_pos.x - 0.5f)); // Pixel to the left
    int y0 = static_cast<int>(std::floor(clamped_pos.y - 0.5f)); // Pixel above
    int x1 = x0 + 1; // Pixel to the right
    int y1 = y0 + 1; // Pixel below

    // Clamp the indices to ensure they are within valid bounds
    x0 = glm::clamp(x0, 0, width - 1);
    y0 = glm::clamp(y0, 0, height - 1);
    x1 = glm::clamp(x1, 0, width - 1);
    y1 = glm::clamp(y1, 0, height - 1);

    // Compute the weights for interpolation
    float dx = clamped_pos.x - (x0 + 0.5f);
    float dy = clamped_pos.y - (y0 + 0.5f);

    float w00 = (1.0f - dx) * (1.0f - dy); 
    float w01 = (1.0f - dx) * dy; 
    float w10 = dx * (1.0f - dy); 
    float w11 = dx * dy; 

    // Access the 4 neighboring pixel values
    T v00 = image.data[y0 * width + x0]; 
    T v01 = image.data[y1 * width + x0]; 
    T v10 = image.data[y0 * width + x1]; 
    T v11 = image.data[y1 * width + x1]; 

    T interpolated_value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

    return interpolated_value;
}



/*
  Core functions.
*/

/// <summary>
/// Applies the bilateral filter on the given disparity image.
/// Ignored pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="size">The kernel size, which is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int size, const float guide_sigma)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // Empty output image.
    auto result = ImageFloat(disparity.width, disparity.height);

    int radius = size / 2;

    // 1. Iterate over all output pixels.
    #pragma omp parallel for 
    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {

            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;

            // 2. Visit all neighboring pixels
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // 3. If a neighbor is outside the image or its disparity == INVALID_VALUE, skip the pixel
                    if (nx < 0 || ny < 0 || nx >= disparity.width || ny >= disparity.height) {
                        continue;
                    }

                    float disparity_center = disparity.data[y * disparity.width + x];
                    float disparity_neighbor = disparity.data[ny * disparity.width + nx];

                    if (disparity_neighbor == INVALID_VALUE) {
                        continue;
                    }

                    glm::vec3 guide_center = guide.data[y * guide.width + x];
                    glm::vec3 guide_neighbor = guide.data[ny * guide.width + nx];

                    // 4. For each neighbor compute w_i = gauss(dist, sigma) * gauss(diff_value, guide_sigma)
                    // dist = euclidean distance between the center and current pixel in pixels
                    float dist = std::sqrt(std::pow((nx - x), 2) + std::pow((ny - y), 2));
                    // diff_value = euclidean distance of the guide image pixel values for the center and current pixel
                    float diff_value = std::sqrt(
                        std::pow(guide_center.r - guide_neighbor.r, 2) + 
                        std::pow(guide_center.g - guide_neighbor.g, 2) + 
                        std::pow(guide_center.b - guide_neighbor.b, 2)
                    );

                    float weight = gauss(dist, sigma) * gauss(diff_value, guide_sigma);

                    weighted_sum += weight * disparity_neighbor;
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0f) {
                result.data[y * disparity.width + x] = weighted_sum / weight_sum;
            } else {
                result.data[y * disparity.width + x] = INVALID_VALUE;
            }
        }
    }

    // Return filtered disparity.
    return result;
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{
    float min = std::numeric_limits<float>::max();
    float max = -std::numeric_limits<float>::max();
    
    // Find minimum and maximum among the VALID image values
    for (int y = 0; y < scalar_image.height; y++) {
        for (int x = 0; x < scalar_image.width; x++) {

            float scalar_value = scalar_image.data[y * scalar_image.width + x];

            if (scalar_value == INVALID_VALUE) {
                continue;
            }

            min = std::min(scalar_value, min);
            max = std::max(scalar_value, max);
        }
    }

    // Lineearly rescale the VALID image values to the [0, 1] range (in-place)
    for (int y = 0; y < scalar_image.height; y++) {
        for (int x = 0; x < scalar_image.width; x++) {
            float& scalar_value = scalar_image.data[y * scalar_image.width + x];

            if (scalar_value == INVALID_VALUE) {
                continue;
            }

            scalar_value = (scalar_value - min) / (max - min);
        }
    }
}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);
        
    for (int y = 0; y < disparity.height; y++) {
        for (int x = 0; x < disparity.width; x++) {
            float disparity_value = disparity.data[y * disparity.width + x];

            // If disparity of a pixel is invalid, set its depth also invalid (INVALID_VALUE)
            if (disparity_value == INVALID_VALUE) {
                depth.data[y * disparity.width + x] = INVALID_VALUE;
            } else {
                //    depth_unscaled = 1.0 / disparity
                depth.data[y * disparity.width + x] = 1.0f / disparity_value;
            }
        }
    }

    // Rescales valid depth values to [0,1] range.
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels 
/// </summary>
/// <param name="depth">Normalized depth image (values in [0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    //
    // Based on physical dimensions, distance, resolution (and hence pixel size) of the screen,
    // as well as physiologically determined distance between viewers pupil (IOD or IPD),
    // compute stereoscopic pixel disparities that will make the display appear at a correct depth
    // represented by linear interpolation between the near and far plane based on the depth input image.
    // 
    // Refer to Lecture 4 for formulas.
    // 
    // Example:
    //    screen distance = 600 mm, near_plane_mm = 550, far_plane == 650, depth = 0.1  
    //         => the pixel should appear 55+0.1(65-55) = 56 cm away from the user
    //         => That is 4 cm in front of the screen.
    //         => That means the pixel disparity will be a negative number ("crossed disparity").
    // 
    // Note:
    //    * All distances are measured orthogonal on the screen and are assumed constant across the screen (ignores the eccentricity variance).
    //    * Invalid pixels (depth==INVALID_VALUE) are to be marked invalid on the output as well.
    //
    
    for (int y = 0; y < depth.height; y++) {
        for (int x = 0; x < depth.width; x++) {
            float depth_value = depth.data[y * depth.width + x];
            
            if (depth_value == INVALID_VALUE) {
                px_disparity.data[y * depth.width + x] = INVALID_VALUE;
                continue;
            }

            // Compute depth in millimeters
            float depth_mm = near_plane_mm + depth_value * (far_plane_mm - near_plane_mm);

            // Calculate disparity in pixels
            float disparity_mm = iod_mm * (screen_distance_mm - depth_mm) / (depth_mm * px_size_mm);
            px_disparity.data[y * depth.width + x] = disparity_mm;

        }
    }

    return px_disparity; // returns disparity measured in pixels
}

/// <summary>
/// Creates a warping grid for an image of specified height and weight.
/// It produces vertex buffer which stores 2D positions of pixel corners,
/// and index buffer which defines triangles by triplets of indices into
/// the vertex buffer (the three vertices form a triangle).
/// 
/// </summary>
/// <param name="width">Image width.</param>
/// <param name="height">Image height.</param>
/// <returns>Mesh, containing a vertex buffer and triangle index buffer.</returns>
Mesh createWarpingGrid(const int width, const int height)
{

    // Build vertex buffer.
    auto num_vertices = (width + 1) * (height + 1);
    auto vertices = std::vector<glm::vec2>(num_vertices);

    //
    // Fill the vertex buffer (vertices) with 2D coordinate of the pixel corners.
    // Expected output coordinates are:
    //   [0,0] for the left top corner of the left top (=first) pixel.
    //   [width,height] for the right bottom corner of the right bottom (=last) pixel.
    //   [0,height] for the left bottom corner of the left bottom pixel.
    // 
    // The order in memory is to be the same as for images (row after row).
    // 
    
    for (int y = 0; y <= height; y++) {
        for (int x = 0; x <= width; x++) {
            vertices[y * (width + 1) + x] = glm::vec2(x, y);
        }
    }

    // Build index buffer.
    auto num_pixels = width * height;
    auto num_triangles = num_pixels * 2;
    auto triangles = std::vector<glm::ivec3>(num_triangles);

    //
    // Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    // Each element of the "triangles" is an integer triplet (glm::ivec3). 
    // It represents a triangle by selecting 3 vertices from the vertex buffer defining its corners in
    // a clockwise manner.
    // We need to fill the index buffer in the same order as pixels are stored in memory (that is row by row)
    // and for each pixel we should generate two triangles that together cover the are of a pixel as follows:
    // 
    //   A ------- B
    //   |  *      |
    //   |    *    |
    //   |      *  |
    //   D ------- C
    // 
    // Where A,B,C,D are the CORNERS of the respective pixel.
    // 
    // For each such pixel, we add two triangles: 
    //     glm::ivec3(A,B,C) and glm::ivec3(A,C,D)  (in this exact order)
    // where A,B,C,D are indices into the vertex buffer.
    // 
    // The result should be a grid that fills an entire image and replaces each pixel with two small triangles.
    //
    
    int triangle_index = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int top_left = y * (width + 1) + x;
            int top_right = top_left + 1;
            int bottom_left = top_left + (width + 1);
            int bottom_right = bottom_left + 1;

            // First triangle A, B, C
            triangles[triangle_index++] = glm::ivec3(top_left, top_right, bottom_right);

            // Second triangle A, C, D
            triangles[triangle_index++] = glm::ivec3(top_left, bottom_right, bottom_left);
        }
    }

    // Combine the vertex and index buffers into a mesh.
    return Mesh { std::move(vertices), std::move(triangles) };
}

/// <summary>
/// Warps a grid based on the given disparity and scaling_factor.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="disparity">Disparity for each PIXEL.</param>
/// <param name="scaling_factor">Global scaling factor for the disparity.</param>
/// <returns>Mesh, the warped grid.</returns>
Mesh warpGrid(Mesh& grid, const ImageFloat& disparity, const float scaling_factor, const BilinearSamplerFloat& sampleBilinear)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float EDGE_EPSILON = 1e-5f * disparity.width;

    //
    // The goal is to modify the x coordinate of the grid vertices based on 
    // the scaled pixel disparity corresponding to the original location of the vertex buffer:
    // 
    // new_grid.vertex.x = grid.vertex.x + scaling_factor * sampled_disparity
    // 
    // where sampled_disparity is a bilinearly interpolated value from the disparity image
    // which we can easily obtained using our other function "sampleBilinear":
    //     sampled_disparity = sampleBilinear(disparity, grid.vertex)
    // 
    // IMPORTANT - in order to keep the grid attached to the image "frame",
    // we must not move the border vertices away => do not move vertices
    // which are within EDGE_EPSILON from the left/right image boundary in the x direction.
    // That means the grid is attached to left and right boundaries but can freely slide along the top/down boundaries.
    //

    // Here is an example use of the bilinear interpolation (using the provided function argument).
    auto interpolated_value = sampleBilinear(disparity, glm::vec2(1.0f, 1.0f));
    // Recommended test: For a 2x2 image it SHOULD return the mean of the 4 pixels.
    
    for (auto& vertex : new_grid.vertices) {
        if (vertex.x < EDGE_EPSILON || vertex.x > disparity.width - EDGE_EPSILON) {
            continue;
        }

        float sampled_disparity = sampleBilinear(disparity, vertex);

        vertex.x += scaling_factor * sampled_disparity;
    }

    return new_grid;
}



/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageWithMask, containing the forward-warped image and a mask image. Mask=1 for valid pixels, Mask=0 for holes</returns>
ImageWithMask forwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);
    
    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_mask = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth mask map with zero.
    std::fill(dst_mask.data.begin(), dst_mask.data.end(), 0.0f);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    // 
    // The goal is to forward warp the image pixels using the disparity displacement provided in 
    // the disparity map along with additional scaling factor. Furthermore, we
    // use depth information to resolve conflicts when multiple pixels attempt
    // to write a single output pixel. 
    //
    // 1. For every input pixel, compute where it should be warped to 
    //    based on the associated disparity and warp_factor. 
    //    Use standard rounding rules to obtain integer position (ie., 0.5 rounds up).
    //      x' = rounding_function(x + disparity * warp_factor)
    //      y' = y
    // 
    // 2. Check the destination depth at the [x',y'] location and compare it with the 
    //    depth of the currently warped pixel (ie., depth[x,y]).
    //
    // 3. If the currently warped pixel has a depth larger or equal to the previous value in the output depth buffer,
    //    stop here and continue with step 1 for the next pixel.
    //
    // 4. Overwrite the output buffers. This means writing:
    //    - the destination image
    //    - the destination depth map
    //    - the mask (mask->1)
    //
    // 
    // Note: Point(s) awarded for a correct and efficient parallel solution using OpenMP.
    //
    
    for (int y = 0; y < src_image.height; y++) {
        for (int x = 0; x < src_image.width; x++) {
            float disparity_value = disparity.data[y * disparity.width + x];
            float depth_value = src_depth.data[y * src_depth.width + x];

            if (disparity_value == INVALID_VALUE || depth_value == INVALID_VALUE) {
                continue;
            }

            // Compute target position
            int target_x = std::round(x + warp_factor * disparity_value);
            int target_y = y;

            if (target_x < 0 || target_x >= src_image.width || target_y < 0 || target_y >= src_image.height) {
                continue; // i think we can skip the target_y checks here no?
            }

            int target_idx = target_y * dst_image.width + target_x;
            if (depth_value >= dst_depth.data[target_idx]) {
                continue;
            }

            dst_image.data[target_idx] = src_image.data[y * src_image.width + x];
            dst_depth.data[target_idx] = depth_value;
            dst_mask.data[target_idx] = 1.0f;
        }
    }


    // Return the warped image.
    return ImageWithMask(dst_image, dst_mask);

}


/// <summary>
/// Applies the gaussian filter on the given image to fill the holes
/// indicated by a binary mask (mask==0 -> missing pixel).
/// Keeps the pixels not marked as holes unchanged.
/// </summary>
/// <param name="img_forward">The image to be filtered and its mask.</param>
/// <param name="size">The kernel size. It is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageRGB, the filtered forward warping image.</returns>
ImageRGB inpaintHoles(const ImageWithMask& img, const int size)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // Rule of thumb for Gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // The output is initialized by a copy of the input image.
    auto result = ImageRGB(img.image);

    const int width = img.image.width;
    const int height = img.image.height;
    const int radius = size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Skip valid pixels (mask >= 0.5).
            if (img.mask.data[y * img.mask.width + x] >= 0.5f) {
                continue;
            }

            float weight_sum = 0.0f;
            glm::vec3 color_sum(0.0f, 0.0f, 0.0f);

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }

                    if (img.mask.data[ny * img.mask.width + nx] < 0.5f) {
                        continue;
                    }

                    float dist = std::sqrt(dx * dx + dy * dy);
                    float weight = std::exp(-(dist * dist) / (2 * sigma * sigma));

                    color_sum += weight * img.image.data[ny * img.image.width + nx];
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0f) {
                result.data[y * result.width + x] = color_sum / weight_sum;
            }
        }
    }

    // Return the inpainted image.
    return result;
}


/// <summary>
/// Backward-warps an image using a warped mesh.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="src_grid">Source grid.</param>
/// <param name="dst_grid">The warped grid.</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>ImageRGB, the backward-warped image.</returns>
ImageRGB backwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const Mesh& src_grid, const Mesh& dst_grid,
    const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{
    assert(src_image.width == src_depth.width && src_image.height == src_depth.height);
    assert(src_grid.triangles.size() == dst_grid.triangles.size());

    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), 1e20f);

    for (size_t tri_idx = 0; tri_idx < dst_grid.triangles.size(); ++tri_idx) {
        const auto& tri_dst = dst_grid.triangles[tri_idx];
        glm::vec2 vert_a_dst = dst_grid.vertices[tri_dst[0]];
        glm::vec2 vert_b_dst = dst_grid.vertices[tri_dst[1]];
        glm::vec2 vert_c_dst = dst_grid.vertices[tri_dst[2]];

        const auto& tri_src = src_grid.triangles[tri_idx];
        glm::vec2 vert_a_src = src_grid.vertices[tri_src[0]];
        glm::vec2 vert_b_src = src_grid.vertices[tri_src[1]];
        glm::vec2 vert_c_src = src_grid.vertices[tri_src[2]];

        float min_x = std::floor(std::min({ vert_a_dst.x, vert_b_dst.x, vert_c_dst.x }));
        float max_x = std::ceil(std::max({ vert_a_dst.x, vert_b_dst.x, vert_c_dst.x }));
        float min_y = std::floor(std::min({ vert_a_dst.y, vert_b_dst.y, vert_c_dst.y }));
        float max_y = std::ceil(std::max({ vert_a_dst.y, vert_b_dst.y, vert_c_dst.y }));

        min_x = std::max(min_x, 0.0f);
        max_x = std::min(max_x, static_cast<float>(dst_image.width - 1));
        min_y = std::max(min_y, 0.0f);
        max_y = std::min(max_y, static_cast<float>(dst_image.height - 1));

        for (int y = static_cast<int>(min_y); y <= static_cast<int>(max_y); ++y) {
            for (int x = static_cast<int>(min_x); x <= static_cast<int>(max_x); ++x) {
                glm::vec2 pt_dst = glm::vec2(x + 0.5f, y + 0.5f);

                if (!isPointInsideTriangle(pt_dst, vert_a_dst, vert_b_dst, vert_c_dst)) {
                    continue;
                }

                glm::vec3 bc = barycentricCoordinates(pt_dst, vert_a_dst, vert_b_dst, vert_c_dst);
                glm::vec2 pt_src = bc.x * vert_a_src + bc.y * vert_b_src + bc.z * vert_c_src;
                float depth_src = sampleBilinear(src_depth, pt_src);
                glm::vec3 color_src = sampleBilinearRGB(src_image, pt_src);

                if (depth_src < dst_depth.data[y * dst_depth.width + x]) {
                    dst_depth.data[y * dst_depth.width + x] = depth_src;
                    dst_image.data[y * dst_image.width + x] = color_src;
                }
            }
        }
    }

    return dst_image;
}



/// <summary>
/// Returns an anaglyph image.
/// </summary>
/// <param name="image_left">left RGB image</param>
/// <param name="image_right">right RGB image</param>
/// <param name="saturation">color saturation to apply</param>
/// <returns>ImageRGB, the anaglyph image.</returns>
ImageRGB createAnaglyph(const ImageRGB& image_left, const ImageRGB& image_right, const float saturation)
{
    // An empty image for the resulting anaglyph.
    auto anaglyph = ImageRGB(image_left.width, image_left.height);

    // 
    // Convert stereoscopic pair into a single anaglyph stereoscopic image
    // for viewing in red-cyan anaglyph glasses.
    // We additionally scale saturation of the image to make the image
    // more "grayscale" since colors are problematic in analglyph image
    // and increase crosstalk (ghosting).
    // 
    // For both left and rigt image:
    // 1. Convert RGB to HSV color space using the provided rgbToHsv() function.
    // 2. Scale the saturation (stored in the second (=Y) component of the vec3) by the "saturation" param.
    // 3. Convert back to RGB using the hsvToRgb().
    // 
    // Combine the two images such that:
    //    * output.red = left.red
    //    * output.green = right.green
    //    * output.blue = right.blue.
    //

    // Example: RGB->HSV->RGB should be approx identity.
    auto rgb_orig = glm::vec3(0.2, 0.6, 0.4);
    auto rgb_should_be_same = hsvToRgb(rgbToHsv(rgb_orig)); // expect rgb == rgb_2 (up to numerical precision)

    for (int y = 0; y < image_left.height; y++) {
        for (int x = 0; x < image_left.width; x++) {
            glm::vec3 left_pixel = image_left.data[y * image_left.width + x];
            glm::vec3 right_pixel = image_right.data[y * image_right.width + x];

            // Adjust saturation
            glm::vec3 left_hsv = rgbToHsv(left_pixel);
            left_hsv.y *= saturation;
            glm::vec3 left_rgb_adjusted = hsvToRgb(left_hsv);

            glm::vec3 right_hsv = rgbToHsv(right_pixel);
            right_hsv.y *= saturation;
            glm::vec3 right_rgb_adjusted = hsvToRgb(right_hsv);

            // Combine the two images
            float red = left_rgb_adjusted.r;
            float green = right_rgb_adjusted.g;
            float blue = right_rgb_adjusted.b;

            anaglyph.data[y * anaglyph.width + x] = glm::vec3(red, green, blue);
        }
    }

    // Returns a single analgyph image.
    return anaglyph;
}


/// <summary>
/// Rotates a grid counter-clockwise around the center by a given angle in degrees.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="center">The center of the rotation (in pixel coords).</param>
/// <param name="angle">Angle in degrees.</param>
/// <returns>Mesh, the rotated grid.</returns>
Mesh rotatedWarpGrid(Mesh& grid, const glm::vec2& center, const float& angle)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float DEGREE2RADIANS = 0.0174532925f;

    //
    // The goal is to rotate the coordinate of the grid vertices
    // counter-clockwise around the 'center' by a given angle in degrees:
    //
    // 1. Create a 3*3 matrix T, there are three steps for this matrix:
    //    Translate from origin to center,
    //    then rotate counter-clockwise by a given angles,
    //    and translate back from center to origin.
    //
    // 2. Multiply each vertex by matrix T to get its new coordinates.
    //

    //
    //    YOUR CODE GOES HERE
    //


    return new_grid;
}



/// <summary>
/// Rotate an image using backward warping based on the provided meshes.
/// </summary>
/// <param name="image">input image</param>
/// <param name="src_grid">original grid</param>
/// <param name="dst_grid">rotated grid</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>rotated image, has the same size as input</returns>
ImageRGB rotateImage(const ImageRGB& image, const Mesh& src_grid, const Mesh& dst_grid, const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{

    //
    // Unused pixels should be black.
    // Pixel that fall outside of the image should be discarded.
    //
    //    YOUR CODE GOES HERE
    //
    return ImageRGB(1, 1); // replace
}