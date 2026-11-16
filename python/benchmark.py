import sys
import time
import numpy as np

# Add build directory to Python path
sys.path.insert(0, "build")
import sobel_ops


def numpy_reference_sobel(img: np.ndarray) -> np.ndarray:
    padded = np.pad(img, pad_width=1, mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def main():
    print(f"Backend active: {'CUDA GPU' if sobel_ops.is_cuda_enabled() else 'CPU Fallback'}")

    height, width = 2048, 2048
    rng = np.random.default_rng(42)
    image = rng.random((height, width), dtype=np.float32)

    # Warmup run (initializes CUDA context on Jetson)
    _ = sobel_ops.sobel_filter(image)

    t0 = time.perf_counter()
    custom_out = sobel_ops.sobel_filter(image)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    ref_out = numpy_reference_sobel(image)
    t3 = time.perf_counter()

    max_diff = np.max(np.abs(custom_out - ref_out))
    print(f"Image size: {width}x{height}")
    print(f"C++/CUDA extension time: {(t1 - t0) * 1000:.2f} ms")
    print(f"NumPy reference time:    {(t3 - t2) * 1000:.2f} ms")
    print(f"Max absolute error:      {max_diff:.6e}")
    assert np.allclose(custom_out, ref_out, atol=1e-4), "Accuracy check failed!"
    print("Verification PASSED!")


if __name__ == "__main__":
    main()
