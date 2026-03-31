#include <opencv2/opencv.hpp>
#include <iostream>
void sobel_cuda(const unsigned char *gray_in, unsigned char *gray_out, int width, int height);
int main(int argc, char **argv)
{
    std::string input = argc > 1 ? argv[1] : "input.jpg";
    cv::Mat img = cv::imread(input, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        return -1;
    cv::Mat out(img.rows, img.cols, CV_8UC1);
    sobel_cuda(img.data, out.data, img.cols, img.rows);
    cv::imwrite("output_edge.jpg", out);
    return 0;
}