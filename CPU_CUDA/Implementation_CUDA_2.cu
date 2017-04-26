#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper/helper_functions.h>
#include <helper/helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
typedef float2 cplx;
#define RANK 1
#define M_PI 3.14159265358979323846
#define FLOAT_MAX 1
struct kernelConf
{
    dim3 block;
    dim3 grid;
};
using namespace std;
using namespace cv;
void readHeaderFile(char *filen, int * dims);
int getCFLSize(char *filen, streampos &size);
int readCflFile(char *file, cplx* data);
void dataScaling(float* data, int size);
int DoDisplayImage2CV(float* dataRaw, float* dataRecons, int xdim, int ydim, int slice, int zipDim, char* filename,int numSlice);
void CUDAgetSpecificSliceData(cplx* input, cplx* outRaw, int spec_slice, int spec_bit, const int* dim, int xdim_new, int ydim_new, int numSlice, int numThread);
void CUDAzeroFilled(cplx* d_inputOneSlice, cplx *d_inputOneSliceNew, int xdim, int ydim, int bitdim, int xdim_new, int ydim_new, int numSlice, int numThread);
__global__ void DoRSSCUDA(cplx* input, float* out, int N, int x, int y, int bit, int numSlice);
__global__ void DoZeroFilledCUDA(cplx* input, cplx* out, int xdim_new, int xdim, int ydim, int bitdim, int numSlice);
__global__ void DoGetSpecificSliceDataCUDA(cplx* input, cplx* outRaw, int xdim_new, int xdim, int ydim, int slicedim, int bitdim, int spec_slice, int numSlice);
__global__ void cufftShift_2D_kernel(cplx* data, int N, int bit, int numSlice);
kernelConf* GenAutoConf_2D(int N);
int DoRECONSOperation( cplx* DataCufftOneSlice, float* out, int *dim, int numSlice, float* times, int numThread);
int DoRSSCUDAOperation(cplx* data, float* out, const int* dim, int size, int spec_slice, int numSlice, int numThread);
int main(int argc, char* argv[])
{
	if(argc < 6)
	{
		cout << "<Usage> <path/filename> <Start_slice> <Number Slices> <New X dimension (Zero Filled Dimension)> <Number Thread (32 or 512)>" << endl;
		return 1;
	}
	clock_t GPU_start1, GPU_end1;
	char *filename = (char*)malloc(100);
	int *dim = (int*)malloc(sizeof(int)*4);
	int *Newdim = (int*)malloc(sizeof(int)*4);
	float *timesRecons = (float*)malloc(sizeof(float)*4);
	sprintf(filename,"%s",argv[1]);
	int slice = atoi(argv[2]);
	int numSlice = atoi(argv[3]);
	int new_x = atoi(argv[4]);
	int numThread = atoi(argv[5]);
	int bit = 8;

	if (new_x < 512)
	{
		cout << "ZIP Dimension Less than 512" << endl;
		new_x = 512;
	}
	else
	{
		int x_comp = 1;
		while(x_comp < new_x)
		{
			x_comp <<= 1;
		}

		if(x_comp > new_x)
		{
			cout << "Zero Filled Dimension Must be Power of TWO" << endl;
			return 1;
		}
	}
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	readHeaderFile(filename, dim);
	int sizeCfl = 1;
	for(int i = 0; i < 4; i++)
	{
		sizeCfl *= dim[i];
		Newdim[i] = dim[i];
	}
	cplx *data ;
	if(prop.canMapHostMemory)
	{
		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
		checkCudaErrors(cudaHostAlloc(&data, sizeof(cplx)*sizeCfl, cudaHostAllocMapped));
	}
	else
	{
		data = (cplx*)malloc(sizeof(cplx)*sizeCfl);
		cout << "Your device is NOT Supporting to using Zero Copy Feature (UVA)" << endl;
		return 1;
	}
	int ret = readCflFile(filename, data);
	if(ret == 1)
	{
		cout << "Error on Reading CFL File" << endl;
		return 1;
	}
	GPU_start1  = clock();
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int xdim_new = new_x;
	int ydim_new = xdim_new;
	int sizeImageNew = xdim_new*ydim_new;
	int sizeImage = xdim*ydim;
	int sizeImageNewSlice = sizeImageNew* bitdim *numSlice;
	int sizeImageNewSliceRaw = sizeImage* bitdim *numSlice;
	size_t nBytes_C = sizeof(cplx)*sizeImageNewSlice;
	size_t nBytes_CRaw = sizeof(cplx)*sizeImageNewSliceRaw;
	size_t nBytes_F = sizeof(float)*sizeImageNew*numSlice;
	size_t nBytes_FRSS = sizeof(float)*sizeImage*numSlice;
	cplx* dataManySliceRaw ;
	if(prop.canMapHostMemory)
	{
		checkCudaErrors(cudaMalloc((void**)&dataManySliceRaw, nBytes_CRaw));
	}
	else
	{
		cout << "Your device is NOT Supporting to using Zero Copy Feature (UVA)" << endl;
	}
	CUDAgetSpecificSliceData(data, dataManySliceRaw, slice, bit, dim, xdim_new, ydim_new, numSlice, numThread);
	if(prop.canMapHostMemory)
	{
		checkCudaErrors(cudaFreeHost(data));
	}
	else
	{
		free(data);
	}

	GPU_end1 = clock();
	float GPUTimer_getspec = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	GPU_start1  = clock();
	cplx* dataManySlice ;
	checkCudaErrors(cudaMalloc((void**)&dataManySlice, nBytes_C));
	CUDAzeroFilled(dataManySliceRaw, dataManySlice, xdim, ydim, bitdim, xdim_new, ydim_new, numSlice, numThread);
	Newdim[0] = xdim_new;
	Newdim[1] = ydim_new;
	xdim = dim[0];
	ydim = dim[1];
	bitdim = dim[3];
	GPU_end1 = clock();
	float GPUTimer_zip = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	GPU_start1  = clock();
	float* dataCUFFT_F ;
	checkCudaErrors(cudaHostAlloc((void**)&dataCUFFT_F, nBytes_F, cudaHostAllocDefault));
	int retCUFFT = DoRECONSOperation(dataManySlice, dataCUFFT_F, Newdim, numSlice, timesRecons, numThread);
	if(retCUFFT == 1)
	{
		cout << "CUFFT Operation is FAILED" << endl;
		return 1;
	}
	GPU_end1 = clock();
	float GPUTimer_fft = timesRecons[0]+timesRecons[1]+timesRecons[2]+timesRecons[3];
	GPU_start1  = clock();
	float *rss;
	checkCudaErrors(cudaHostAlloc((void**)&rss, nBytes_FRSS, cudaHostAllocDefault));
	int retRSS = DoRSSCUDAOperation(dataManySliceRaw, rss, dim, sizeCfl, slice, numSlice, numThread);
	if(retRSS == 1)
	{
		cout << "RSS Operation is FAILED" << endl;
		return 1;
	}
	GPU_end1 = clock();
	float GPUTimer_RSSRaw = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	float GPUTimer = GPUTimer_getspec + GPUTimer_zip + GPUTimer_fft + GPUTimer_RSSRaw;
	cout << "==========================================================================================================" << endl;
	cout << "<path/filename> <Start Slice> <Number Slice Per Operatio> <New X dimension (Zero Filled Dimension)> <Number Thread (32 or 512>" << endl;
	cout << "<kspace> " << std::to_string(slice) << " " << std::to_string(numSlice) << " " << std::to_string(new_x) << " " << std::to_string(numThread)<< endl;
	cout << " " << endl;
	cout << "Timing <GetSpec> <ZIP> <FFT> <SHIFT> <RSS> <SCALING> <RSS RAW> <TOTAL>" << endl;
	cout << "Timing " << std::to_string(GPUTimer_getspec*1000) << " " << std::to_string(GPUTimer_zip*1000) << " " << std::to_string(timesRecons[0]*1000) << " " << std::to_string(timesRecons[1]*1000)<<" " << std::to_string(timesRecons[2]*1000) << " " << std::to_string(timesRecons[3]*1000) << " " <<std::to_string(GPUTimer_RSSRaw*1000) << " " << std::to_string(GPUTimer*1000)<< endl;
	cout << "==========================================================================================================" << endl;
	/*int retCV = DoDisplayImage2CV(rss, dataCUFFT_F, xdim, ydim, slice, new_x, filename, numSlice);
	if(retCV == 1)
	{
		cout << "Display Image is Failed" << endl;
		return 1;
	}*/
	free(filename);
	free(Newdim);
	free(dim);
	free(timesRecons);
	checkCudaErrors(cudaFree(dataManySliceRaw));
	checkCudaErrors(cudaFree(dataManySlice));
	checkCudaErrors(cudaFreeHost(dataCUFFT_F));
	checkCudaErrors(cudaFreeHost(rss));
	checkCudaErrors(cudaDeviceReset());
	return 0;
}

void readHeaderFile(char *filen, int * dims)
{
	char path[20];
	sprintf(path,"%s.hdr",filen);
	FILE *myFile;
	myFile = fopen(path,"r");
	string line;
	streampos size;
	
	fseek(myFile, 13, SEEK_SET);
	for(int i = 0; i < 4; i++)
	{
		fscanf(myFile,"%d",&dims[i]);
	}
}
int readCflFile(char *filen, cplx* data)
{
	// cout << "Read CFL File" << endl;
	streampos size;
	char path[20];
	sprintf(path,"%s.cfl",filen);
	ifstream file(path, ios::in | ios::binary | ios::ate);
	if(file.is_open())
	{
		size = file.tellg();
		if(size)
		{
			// cout << "Read CFL file Success" << endl;
		}
		else
		{
			cout << "Error Allocating Memory" << endl;
			return 1;
		}
	}
	else
	{
		cout << "Unable to open file";
		return 1; 
	}

	if(file.is_open())
	{
		file.seekg(0, ios::beg);
		file.read((char*)data, size);
		file.close();
	}
	return 0;
}
void dataScaling(float* data, int size)
{
	float max = 0;
	for(int i = 0; i< size; i++)
	{
		if(data[i] > max)
		{
			max = data[i];
		}
	}

	for(int j = 0; j < size; j++)
	{
		data[j] = data[j]/max;
	}
}
void CUDAzeroFilled(cplx* d_inputOneSlice, cplx *d_inputOneSliceNew, int xdim, int ydim, int bitdim, int xdim_new, int ydim_new, int numSlice, int numThread)
{
	dim3 tpb(xdim, ydim);
	dim3 bpg(1,1);
	if(xdim > numThread)
	{
		tpb.x = numThread;
		tpb.y = numThread;
		bpg.x = ceil(double(xdim)/double(tpb.x));
		bpg.y = ceil(double(xdim)/double(tpb.y));
	}

	DoZeroFilledCUDA<<<tpb, bpg>>>(d_inputOneSlice, d_inputOneSliceNew, xdim_new, xdim, ydim, bitdim, numSlice);
	checkCudaErrors(cudaDeviceSynchronize());
}
__global__ void DoZeroFilledCUDA(cplx* input, cplx* out, int xdim_new, int xdim, int ydim, int bitdim, int numSlice)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int indexOne = idy * xdim + idx;
	int N = xdim*ydim;
	int residue = (xdim_new-xdim)/2;
	int sidx = (xdim_new*residue)+residue;
	int offset = 2*residue;
	int offsetPerBitmask = xdim_new*xdim_new;
	int offsetBitmask = N*numSlice;
	int offsetSlice = N;
	int offsetPerSlice = xdim_new*xdim_new;

	if(idx < xdim && idy < ydim)
	{
		for(int m = 0; m < bitdim; m++)
		{
			for(int l = 0; l < numSlice; l++)
			{
				out[sidx + indexOne + (idy*offset) + (l*offsetPerSlice) + (m*offsetPerBitmask)] = input[indexOne + (l*offsetSlice) + (m*offsetBitmask)];
			}
		}
	}
}
void CUDAgetSpecificSliceData(cplx* input, cplx* d_outRaw, int spec_slice, int spec_bit, const int* dim, int xdim_new, int ydim_new, int numSlice, int numThread)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int slicedim = dim[2];
	int bitdim = dim[3];
	cplx* d_input;
	checkCudaErrors(cudaHostGetDevicePointer(&d_input, input, 0));

	dim3 tpb(xdim, ydim);
	dim3 bpg(1,1);
	if(xdim > numThread)
	{
		tpb.x = numThread;
		tpb.y = numThread;
		bpg.x = ceil(double(xdim)/double(tpb.x));
		bpg.y = ceil(double(xdim)/double(tpb.y));
	}
	DoGetSpecificSliceDataCUDA<<<tpb, bpg>>>(d_input, d_outRaw, xdim_new, xdim, ydim, slicedim, bitdim, spec_slice, numSlice);
	checkCudaErrors(cudaDeviceSynchronize());
}
__global__ void DoGetSpecificSliceDataCUDA(cplx* input, cplx* out, int xdim_new, int xdim, int ydim, int slicedim, int bitdim, int spec_slice, int numSlice)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int indexOne = idy * xdim + idx;
	int N = xdim*ydim;
	int offsetBitmask = N*numSlice;
	int offsetSlice = N;
	int sliceP = 0;
	if(idx < xdim && idy < ydim)
	{
		for(int m = 0; m < bitdim; m++)
		{
			sliceP = 0;
			for(int l = spec_slice; l < (spec_slice+numSlice); l++)
			{
				out[indexOne + (sliceP*offsetSlice) + (m*offsetBitmask)] = input[indexOne + (l*N) + (m*slicedim*N)];
				sliceP++;
			}
		}
	}
}
__global__ void DoRSSCUDA(cplx* input, float* out, int N, int x, int y, int bit, int numSlice)
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	float temp;
	float sum;
	int sizeManySlice = N*numSlice;
	if(tidx < N) 
	{
		for(int j = 0; j < numSlice; j++)
		{
			temp = 0;
			sum = 0;
			for(int i = 0; i < bit; i++)
			{
				temp = cuCabsf(input[tidx+(j*N)+(i*sizeManySlice)]);
				temp = (float)pow((double)temp, (double)2);
				sum += temp;
			}
			sum = (float)sqrt(sum);
			out[tidx+(j*N)] = sum;
		}
	}
}
int DoDisplayImage2CV(float* dataRaw, float* dataRecons, int xdim, int ydim, int spec_slice, int zipdimx, char* filename, int numSlice)
{
	char winNameRaw[100];
	char winNameRecons[100];
	cv::Mat imgRaw;
	cv::Mat imgRecons;
	int oneDimRaw = xdim*ydim;
	int oneDim = zipdimx*zipdimx;
	size_t nBytes_One = sizeof(float)*oneDim;
	size_t nBytes_OneRaw = sizeof(float)*oneDimRaw;
	float* oneImage = (float*)malloc(nBytes_One);
	float* oneImageRaw = (float*)malloc(nBytes_OneRaw);
	int offset = 0;
	int offsetRaw = 0;

	for(int a = 0; a < numSlice; a++)
	{
		offset = a*oneDim;
		offsetRaw = a*oneDimRaw;
		memcpy(oneImage, dataRecons + offset, nBytes_One);
		memcpy(oneImageRaw, dataRaw + offsetRaw, nBytes_OneRaw);
		imgRaw = cv::Mat(xdim, ydim, CV_32F, oneImageRaw);
		imgRecons = cv::Mat(zipdimx, zipdimx, CV_32F, oneImage);
		if(imgRaw.rows == 0 || imgRaw.cols == 0)
			return 1;
		if(imgRecons.rows == 0 || imgRecons.cols == 0)
			return 1;
		sprintf(winNameRecons,"Reconstructed Image on CV - Slice %d",spec_slice+a );
		sprintf(winNameRaw,"Raw Image on CV - Slice %d",spec_slice+a );
		cv::namedWindow(winNameRaw, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
		cv::namedWindow(winNameRecons, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
		cv::imshow(winNameRaw,imgRaw);
		cv::waitKey(500);
		cv::imshow(winNameRecons,imgRecons);
		cv::waitKey(1500);
	}
	cv::waitKey();
	free(oneImage);
	free(oneImageRaw);
	
	return 0;
}
int DoRECONSOperation( cplx* d_data, float* out, int *dim, int numSlice, float* times, int numThread)
{
	clock_t start, end;
	start = clock();
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int sizeOneImage = xdim*ydim;
	size_t nBytes_C = sizeof(cplx)*sizeOneImage;
	size_t nBytes_F = sizeof(float)*sizeOneImage*numSlice;
	float* d_out;
	cplx* d_temp;
	checkCudaErrors(cudaMalloc((void**)&d_out, nBytes_F));
	checkCudaErrors(cudaMalloc((void**)&d_temp, nBytes_C));
	int offset = 0;
	cufftHandle plan;
	int batch = 1;
    int nn[RANK] = {sizeOneImage};
    cufftPlanMany(&plan, RANK, nn, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, batch);
    for(int a = 0; a < numSlice*bitdim; a++)
	{
		offset = a*sizeOneImage;
		checkCudaErrors(cudaMemcpy(d_temp, d_data+offset, nBytes_C, cudaMemcpyDeviceToDevice));
		cufftExecC2C(plan, d_temp, d_temp, CUFFT_INVERSE);
		checkCudaErrors(cudaMemcpy(d_data+offset, d_temp, nBytes_C, cudaMemcpyDeviceToDevice));
	}
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_temp));
    end = clock();
    times[0] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock();
	if(xdim != ydim)
	{
		cout << "X Dimension Must be same with YDIM" << endl;
		return 1;
	}
	else
	{
		const int N = xdim;
        kernelConf* conf = GenAutoConf_2D(N);
        cufftShift_2D_kernel <<< conf->grid, conf->block >>> (d_data, N, bitdim, numSlice); // In-Place
        checkCudaErrors(cudaDeviceSynchronize());
	}
	end = clock();
    times[1] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock();
	dim3 threadsPerBlock(sizeOneImage);
	dim3 blocksPerGrid(1);
	if(sizeOneImage > numThread)
	{
		threadsPerBlock.x = numThread;
		blocksPerGrid.x = ceil(double(sizeOneImage)/double(threadsPerBlock.x));
	}
	DoRSSCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_out, sizeOneImage, xdim, ydim, bitdim, numSlice);
	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
    times[2] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock();
	int sizeManySliceWOBit = sizeOneImage*numSlice;
	checkCudaErrors(cudaMemcpy(out, d_out, nBytes_F, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_out));
	cufftDestroy(plan);
	dataScaling(out, sizeManySliceWOBit);
	end = clock();
    times[3] = (float)(end-start)/CLOCKS_PER_SEC;
	return 0;
}

kernelConf* GenAutoConf_2D(int N)
{
    kernelConf* autoConf = (kernelConf*) malloc(sizeof(kernelConf));

    int threadsPerBlock_X;
    int threadsPerBlock_Y;

    if (2 <= N && N < 4)
    {
        threadsPerBlock_X = 2;
        threadsPerBlock_Y = 2;
    }
    if (4 <= N && N < 8)
    {
        threadsPerBlock_X = 4;
        threadsPerBlock_Y = 4;
    }
    if (8 <= N && N < 16)
    {
        threadsPerBlock_X = 8;
        threadsPerBlock_Y = 8;
    }
    if (16 <= N && N < 32)
    {
        threadsPerBlock_X = 16;
        threadsPerBlock_Y = 16;
    }
    if (N >= 32)
    {
        threadsPerBlock_X = 32;
        threadsPerBlock_Y = 32;
    }

    autoConf->block = dim3(threadsPerBlock_X, threadsPerBlock_Y, 1);
    autoConf->grid = dim3((N / threadsPerBlock_X), (N / threadsPerBlock_Y), 1);
    return autoConf;
}

__global__
void cufftShift_2D_kernel(cplx* data, int N, int bit, int numSlice)
{
    int sLine = N;
    int sSlice = N * N;
    int sManySlice = N * N*numSlice;
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    int index = (yIndex * N) + xIndex;
    cplx regTemp;
    if (xIndex < N / 2)
    {
        if (yIndex < N / 2)
        {
        	for(int i = 0; i<bit; i++)
        	{
        		for(int j = 0; j < numSlice; j++)
        		{
        			regTemp = data[index + (j*sSlice) + (i*sManySlice)]; // I am add "+ (i*sSlice)"
        			data[index+ (j*sSlice) + (i*sManySlice)] = data[index + sEq1 + (j*sSlice) + (i*sManySlice)]; // I am add "+ (i*sSlice)"
        			data[index + sEq1+ (j*sSlice) + (i*sManySlice)] = regTemp; // I am add "+ (i*sSlice)"
        		}
        	}
        }
    }
    else
    {
        if (yIndex < N / 2)
        {
        	for(int i = 0; i< bit; i++)
        	{
        		for(int j = 0; j < numSlice; j++)
        		{
	        		regTemp = data[index + (j*sSlice) + (i*sManySlice)]; // I am add "+ (i*sSlice)"
	        		data[index + (j*sSlice) + (i*sManySlice)] = data[index + sEq2 + (j*sSlice) + (i*sManySlice)]; // I am add "+ (i*sSlice)"
	        		data[index + sEq2 + (j*sSlice) + (i*sManySlice)] = regTemp; // I am add "+ (i*sSlice)"
	        	}
        	}
        }
    }
}
int DoRSSCUDAOperation(cplx* d_data, float* out, const int* dim, int size, int spec_slice, int numSlice, int numThread)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int sizeOneImg = xdim*ydim;
	size_t nBytes_F = sizeof(float)*sizeOneImg*numSlice;
	float* d_out;
	checkCudaErrors(cudaMalloc((void**)&d_out, nBytes_F));
	if(xdim != ydim)
	{
		cout << "X Dimension Must be same with YDIM" << endl;
		return 1;
	}
	dim3 threadsPerBlock(sizeOneImg);
	dim3 blocksPerGrid(1);
	if(sizeOneImg > numThread)
	{
		threadsPerBlock.x = numThread;
		blocksPerGrid.x = ceil(double(sizeOneImg)/double(threadsPerBlock.x));
	}
	DoRSSCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_out, sizeOneImg, xdim, ydim, bitdim, numSlice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(out, d_out, nBytes_F, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_out));
	return 0;
}