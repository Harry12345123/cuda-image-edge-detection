#include <cuda_runtime.h>
#include <math.h>
__device__ int clamp_xy(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__global__ void sobel_kernel(const unsigned char *in, unsigned char *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int sumx = 0, sumy = 0;
    for (int ky = -1; ky <= 1; ++ky)
    {
        for (int kx = -1; kx <= 1; ++kx)
        {
            int nx = clamp_xy(x + kx, 0, w - 1);
            int ny = clamp_xy(y + ky, 0, h - 1);
            unsigned char p = in[ny * w + nx];
            sumx += gx[ky + 1][kx + 1] * p;
            sumy += gy[ky + 1][kx + 1] * p;
        }
    }
    int mag = (int)sqrtf((float)(sumx * sumx + sumy * sumy));
    out[y * w + x] = (unsigned char)(mag > 255 ? 255 : mag);
}
void sobel_cuda(const unsigned char *gray_in, unsigned char *gray_out, int width, int height)
{
    size_t bytes = width * height;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, gray_in, bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    sobel_kernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(gray_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}