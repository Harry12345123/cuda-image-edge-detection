# CUDA Image Edge Detection

A CUDA C++ project for edge detection on the GPU.

This repository is a practical starting point for developers interested in GPU-based computer vision and image analysis.

## Features

- edge detection with CUDA
- GPU-accelerated image filtering
- beginner-friendly C++ code structure
- useful visual output for learning

## Tech Stack

- C++
- CUDA
- OpenCV

## Project Goal

This project helps developers understand:

- how edge detection works in image processing
- how to accelerate filtering operations with CUDA
- how GPU parallelism applies to computer vision basics

## Future Improvements

- Sobel operator optimization
- Canny edge version
- video edge detection
- Jetson real-time demo

## Related Topics

CUDA, Edge Detection, Computer Vision, GPU Image Processing, C++, OpenCV

## Author

Harry12345123

## More

This project is part of my CUDA visual computing learning series.

## Requirements

Before building this project, make sure your system has:

- CUDA Toolkit
- OpenCV
- CMake 3.18 or later
- C++17 compatible compiler

## Build
Use the following commands to compile the project:

```bash
mkdir build
cd build
cmake ..
make -j

Run

After building, run the program with:

./cuda_image_edge_detection input.jpg

Notes
Make sure input.jpg exists in the project root directory
You can replace the input file with your own image
The executable name depends on your CMake project configuration
