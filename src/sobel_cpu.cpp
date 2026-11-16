#include "sobel.h"
#include <algorithm>
#include <cmath>

void compute_sobel(const float* input, float* output, int width, int height) {
    auto at = [&](int r, int c) -> float {
        r = std::min(std::max(r, 0), height - 1);
        c = std::min(std::max(c, 0), width - 1);
        return input[r * width + c];
    };

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float gx = -at(r - 1, c - 1) + at(r - 1, c + 1)
                       - 2.0f * at(r, c - 1) + 2.0f * at(r, c + 1)
                       - at(r + 1, c - 1) + at(r + 1, c + 1);

            float gy = -at(r - 1, c - 1) - 2.0f * at(r - 1, c) - at(r - 1, c + 1)
                       + at(r + 1, c - 1) + 2.0f * at(r + 1, c) + at(r + 1, c + 1);

            output[r * width + c] = std::sqrt(gx * gx + gy * gy);
        }
    }
}

bool is_cuda_enabled() {
    return false;
}
