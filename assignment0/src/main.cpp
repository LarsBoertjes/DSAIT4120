#include "your_code_here.h"

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

static const int NUM_TASKS = 2000;

/// <summary>
/// Main method. Runs default tests. Feel free to modify it, add more tests and experiments,
/// change the input images etc. The file is not part of the solution. All solutions have to 
/// implemented in "your_code_here.h".
/// </summary>
/// <returns>0</returns>
int main()
{
    std::chrono::steady_clock::time_point time_start, time_end;
    // Print information about OpenMP.
    printOpenMPStatus();
    
    // 0. Load inputs from files.
    auto image = ImageRGB(dataDirPath / "mandrill.jpg");
    image.writeToFile(outDirPath / "input.png");

    // 1. Scale image as a copy.
    auto scaled_copy = scaleImageCopy(image, 0.5f);
    scaled_copy.writeToFile(outDirPath / "scaled_copy.png");

    // 2. Scale the input itself (in-place) without making a copy.
    scaleImageInPlace(image, 0.5f);
    image.writeToFile(outDirPath / "scaled_inplace.png");

    // 3. Test parallel loop.
    std::cout << "Starting performance tests..." << std::endl; 
    time_start = std::chrono::steady_clock::now();
    doLotOfWork(NUM_TASKS);
    time_end = std::chrono::steady_clock::now();
    std::cout << "The " << NUM_TASKS  << " tasks took "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms to complete." << std::endl;

    // Vector with random values.
    std::srand(unsigned(4548421215)); // constant seed => every run the same
    //std::srand(unsigned(std::time(nullptr))); // random seed => changes every run
    std::vector<float> random_vector(NUM_TASKS);
    std::generate(random_vector.begin(), random_vector.end(), []() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    // 4. Sum the array.
    time_start = std::chrono::steady_clock::now();
    auto sum = getSum(random_vector);
    time_end = std::chrono::steady_clock::now();
    std::cout << "The sum = " << sum << " was found after "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms." << std::endl;

    // 5. Find minimum value.
    time_start = std::chrono::steady_clock::now();
    auto min_val = getMinimumValue(random_vector);
    time_end = std::chrono::steady_clock::now();
    std::cout << "The min value = " << min_val << " was found after "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms." << std::endl;

    // Return 0 to tell OS all went well.
    std::cout << "All done!" << std::endl;
    return 0;
}
