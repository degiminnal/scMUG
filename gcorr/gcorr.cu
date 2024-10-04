/*
 * Author: LDM
 * Date: 2023.08.18
 * Version: 1.0
 * Description: Using GPU to calculate PCC of single cell expression data.
 * Contact: degim@tju.edu.cn
 */

#include <cuda.h>
#include <iostream>

using namespace std;

#define M 9440

#define get_bid() (blockIdx.x)
#define get_tid() (threadIdx.x)

extern "C" __declspec(dllexport) int get_m(int&);
extern "C" __declspec(dllexport) int pearson(int, int, float*, float*, float*, float*, float*, float*, float*);


int m, n;                  // m: gene number (m<56640)   n: cell number
float *a;                  // Single cell expression data array stored in CPU，shape m*n
float *p;                  // Single cell expression data array stored in GPU
float *dstds;              // The standard deviation of the genes stored in the GPU
float *dcorr0;             // Partition 0 of calculation results stored in the GPU
float *dcorr1;             // Partition 1 of calculation results stored in the GPU
float *dcorr2;             // Partition 2 of calculation results stored in the GPU
float *dcorr3;             // Partition 3 of calculation results stored in the GPU
float *dcorr4;             // Partition 4 of calculation results stored in the GPU
float *dcorr5;             // Partition 5 of calculation results stored in the GPU
float *corr_buf;           // The buffer of the CPU that stores calculation results
size_t pitch_p, pitch_c;   // GPU inter-row address distance


__device__ void calc(volatile float* sdata, int tid, float* dx, float* dy, int n)
{
    sdata[tid] = 0;
    for(int idx=tid; idx<n; idx+=32){
        sdata[tid] += dx[idx] * dy[idx];
    }
    for(int t=16;t>=1;t/=2){
        sdata[tid] += sdata[tid+t];
    }
}

__device__ float* getptr(int r, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch)
{
    int a=r/M, b=r%M;
    if(a==0) return (float*)((char*)dcorr0 + b * pitch );
    if(a==1) return (float*)((char*)dcorr1 + b * pitch );
    if(a==2) return (float*)((char*)dcorr2 + b * pitch );
    if(a==3) return (float*)((char*)dcorr3 + b * pitch );
    if(a==4) return (float*)((char*)dcorr4 + b * pitch );
    if(a==5) return (float*)((char*)dcorr5 + b * pitch );
    return nullptr;
}

__global__ void add(int m, int n, float *p, int pitch_p, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch_c, float *dstds)
{
    __shared__ float sdata[64];
    int tid = get_tid(), bid = get_bid();

    float *dx, *dy;
    for(int i=bid;i<(m+1)/2;i+=gridDim.x){
        dx = (float*)((char*)p + i * pitch_p);
        for(int j=0;j<=i;j++){
            dy = (float*)((char*)p + j * pitch_p);
            calc(sdata, tid, dx, dy, n);
            if(tid==0) getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j]=sdata[0];
        }        
        dx = (float*)((char*)p + (m - 1 - i) * pitch_p);
        for(int j=0;j<m-i;j++){
            dy = (float*)((char*)p + j * pitch_p);
            calc(sdata, tid, dx, dy, n);
            if(tid==0) getptr(m-1-i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j]=sdata[0];
        }
        if(tid==0) {
            dstds[i] = sqrt(getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[i]);
            dstds[m-1-i] = sqrt(getptr(m-1-i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[m-1-i]);
        }
    }
}

__global__ void div(int m, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch_c, float *dstds)
{
    int bid = get_bid();
    for(int i=bid;i<m;i+=gridDim.x){
        for(int j=0;j<=i;j++){
            getptr(j, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[i] =
            getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j] /= (dstds[i] * dstds[j]);
        }
    }
}

extern "C" {
    int get_m(int& _m){_m=M;return 0;}
    int pearson(int _m, int _n, float* _a, float* _corr0, float* _corr1, float* _corr2, float* _corr3, float* _corr4, float* _corr5) {
        m = _m, n = _n, a = _a;

        for(int i=0;i<m;i++){                                                         
            float sum = 0;
            for(int j=0;j<n;j++){
                sum += a[i*n+j];
            }
            float mean = sum / n;
            for(int j=0;j<n;j++){
                a[i*n+j] -= mean;
            }
        }

        cudaMalloc((void**)&dstds, sizeof(float)*m);                                    
        cudaMallocPitch((void**)&p, &pitch_p, sizeof(float)*n, m);                   
        cudaMallocHost((void**)&corr_buf, sizeof(float)*m*M);                         
        if(m>0)   cudaMallocPitch((void**)&dcorr0, &pitch_c, sizeof(float)*m, M);       
        if(m>1*M) cudaMallocPitch((void**)&dcorr1, &pitch_c, sizeof(float)*m, M);      
        if(m>2*M) cudaMallocPitch((void**)&dcorr2, &pitch_c, sizeof(float)*m, M);      
        if(m>3*M) cudaMallocPitch((void**)&dcorr3, &pitch_c, sizeof(float)*m, M);     
        if(m>4*M) cudaMallocPitch((void**)&dcorr4, &pitch_c, sizeof(float)*m, M);     
        if(m>5*M) cudaMallocPitch((void**)&dcorr5, &pitch_c, sizeof(float)*m, M);     

        cudaMemcpy2D(p, pitch_p, a, sizeof(float)*n, sizeof(float)*n, m, cudaMemcpyHostToDevice);
        
        dim3 ts(32), bs(1344);
        add<<<bs, ts>>>(m, n, p, pitch_p, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds); 
        div<<<bs, ts>>>(m, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds); 
        cudaDeviceSynchronize();

        if(m>0){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr0, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr0, corr_buf, sizeof(float)*m*M); 
        }
        if(m>1*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr1, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr1, corr_buf, sizeof(float)*m*M); 
        }
        if(m>2*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr2, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr2, corr_buf, sizeof(float)*m*M); 
        }
        if(m>3*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr3, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr3, corr_buf, sizeof(float)*m*M); 
        }
        if(m>4*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr4, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr4, corr_buf, sizeof(float)*m*M); 
        }
        if(m>5*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr5, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr5, corr_buf, sizeof(float)*m*M); 
        }

        cudaFree(p);                                                                                 
        cudaFree(dstds);
        cudaFreeHost(corr_buf);
        if(m>0)   cudaFree(dcorr0);                                            
        if(m>1*M) cudaFree(dcorr1);                                            
        if(m>2*M) cudaFree(dcorr2);                                            
        if(m>3*M) cudaFree(dcorr3);                                            
        if(m>4*M) cudaFree(dcorr4);                                            
        if(m>5*M) cudaFree(dcorr5);  

        return 0;
    }
}
