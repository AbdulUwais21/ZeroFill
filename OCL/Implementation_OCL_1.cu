#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <clFFT.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define RANK 1
#define M_PI 3.14159265358979323846
#define fftshift(out, in, x, y) circshift(out, in, x, y, (x/2), (y/2))
struct kernelConf
{
    dim3 block;
    dim3 grid;
};
using namespace std;
using namespace cv;
void readHeaderFile(char *filen, int * dims);
int readCflFile(char *file, float2* data);
int DoDisplayImage2CV(float* dataRaw, float* dataRecons, int xdim, int ydim,  int spec_slice, int zipdimx, char* filename, int numSlice);
const char *getErrorString(cl_int error);
bool checkSuccess(cl_int errorNumber);
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);
cl_device_id create_device();
bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program);
int OCLgetSpecificSliceData(cl_mem data, cl_mem out, int* dim,  int spec_slice, int numSlice, int xdim_new, int ydim_new, cl_command_queue commandQueue, cl_program program, int numThread);
int OCLZeroFilled(cl_mem d_inputOneSlice, cl_mem d_inputOneSliceNew, int xdim, int ydim, int bitdim, int numSlice, int xdim_new, int ydim_new, cl_command_queue commandQueue, cl_program program, int numThread);
int OCL_fftshift(cl_mem inputData, int xdim, int ydim, int bitdim, int numSLice, cl_command_queue commandQueue, cl_program program, int numThread);
int DoRECONSOperation( cl_mem inputData, float* d_out, int *dim, int numSlice, cl_context context, cl_command_queue commandQueue, cl_program program, float* times, int numThread);
int OCL_clfft(cl_mem inputData, int xdim, int ydim, int bitdim, int numSLice, cl_context context, cl_command_queue commandQueue);
int OCL_dataScaling(cl_mem d_out, int sizeOneImage, cl_context context, cl_command_queue commandQueue, cl_program program, int numThread);
int OCL_RSS(cl_mem inputData, cl_mem outData, int sizeOneImage, int xdim, int ydim, int bitdim, int numSLice, cl_command_queue commandQueue, cl_program program, int numThread);
int OCL_dataScaling(cl_mem d_out, int sizeOneImage, cl_command_queue commandQueue, cl_program program, int numThread);
int OCLMax_reduce(cl_mem d_out, cl_mem max, int sizeOneImage, cl_command_queue commandQueue, cl_program program, int numThread);
int OCLParallel_devide(cl_mem d_out, cl_mem max, int sizeOneImage, cl_command_queue commandQueue, cl_program program, int numThread);
int DoRSSOCLOperation(cl_mem inputData, float* output, const int* dim, int size, int spec_slice, int numSlice, cl_context  context, cl_command_queue commandQueue, cl_program program, int numThread);
int main(int argc, char* argv[])
{
	if(argc < 6)
	{
		cout << "<Usage> <path/filename> <Spec_slice> <Number Slice(s)> <New X dimension (Zero Filled Dimension)> <Number Thread (32 atau 512)>" << endl;
		return 1;
	}
	cout << "========================================================================================================================" << endl;
	clock_t GPU_start1, GPU_end1;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	const unsigned int numberMemoryObjects = 3;
	cl_mem memObj_data[numberMemoryObjects] = {0,0,0};
	int err;
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}
	commandQueue = clCreateCommandQueue(context, device, 0, &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}
	if(!createProgram(context, device, "device_code/Recons.cl", &program))
	{
		cerr << "Failed to create a program OpenCL" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	char *filename = (char*)malloc(100);
	int *dim = (int*)malloc(sizeof(int)*4);
	int *Newdim = (int*)malloc(sizeof(int)*4);
	float *timesRecons = (float*)malloc(sizeof(float)*4);
	sprintf(filename,"%s",argv[1]);
	int slice = atoi(argv[2]);
	int numSlice = atoi(argv[3]);
	int new_x = atoi(argv[4]);
	int numThread = atoi(argv[5]);
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
	readHeaderFile(filename, dim);
	int sizeCfl = 1;
	for(int i = 0; i < 4; i++)
	{
		sizeCfl *= dim[i];
		Newdim[i] = dim[i];
	}
	size_t buffer_size = sizeCfl * sizeof(float2);
	float2 *data = (float2*)malloc(buffer_size);
	int ret = readCflFile(filename, data);
	if(ret == 1)
	{
		cout << "Error on Reading CFL File" << endl;
		return 1;
	}
	memObj_data[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}
	err = clEnqueueWriteBuffer(commandQueue, memObj_data[0], CL_TRUE, 0, buffer_size, data, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	free(data);
	GPU_start1  = clock();
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int xdim_new = new_x;
	int ydim_new = xdim_new;
	int sizeImageNew = xdim_new*ydim_new;
	int sizeImage = xdim*ydim;
	int sizeImageNewSlice = sizeImageNew* bitdim * numSlice;
	int sizeImageNewSliceRaw = sizeImage* bitdim * numSlice;
	size_t nBytes_CNew = sizeof(float2)*sizeImageNewSlice;
	size_t nBytes_C = sizeof(float2)*sizeImageNewSliceRaw;
	size_t nBytes_F = sizeof(float)*sizeImageNew*numSlice;
	size_t nBytes_FRSS = sizeof(float)*sizeImage*numSlice;
	memObj_data[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes_C, NULL, &err);
	checkSuccess(err);
	float2* dataManySlice = (float2*)malloc(nBytes_C);
	err = OCLgetSpecificSliceData(memObj_data[0], memObj_data[1], dim, slice, numSlice, xdim_new, ydim_new, commandQueue, program, numThread);
	if(err == 1)
	{
		cerr << "Failed to Get Specific Slice(s) Data" << __FILE__ << ":" << __LINE__ << endl;
		exit(1);
	}	
	clReleaseMemObject(memObj_data[0]);
	free(dataManySlice);
	GPU_end1 = clock();
	float GPUTimer_getspec = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	GPU_start1  = clock();
	memObj_data[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes_CNew, NULL, &err);
	checkSuccess(err);
	cl_uint initValue = 0;
	err = clEnqueueFillBuffer(commandQueue, memObj_data[2], &initValue, sizeof(cl_uint), 0, nBytes_CNew, 0, NULL, NULL);
	clFinish(commandQueue);
	if(!checkSuccess(err))
	{
		cerr << "Failed to Fill Buffer " << __FILE__ << ":" << __LINE__ << endl;
		cerr << getErrorString(err) << endl;
		exit(1);
	}
	err = OCLZeroFilled(memObj_data[1], memObj_data[2], xdim, ydim, bitdim, numSlice, xdim_new, ydim_new, commandQueue, program, numThread);
	clFinish(commandQueue);
	if(err == 1)
	{
		cerr << "Failed to do Zero Filled - Interpolation" << __FILE__ << ":" << __LINE__ << endl;
		exit(1);
	}
	Newdim[0] = xdim_new;
	Newdim[1] = ydim_new;
	xdim = Newdim[0];
	ydim = Newdim[1];
	bitdim = Newdim[3];
	GPU_end1 = clock();
	float GPUTimer_zeroFill = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	GPU_start1  = clock();
	float* dataCUFFT_F = (float*)malloc(nBytes_F);
	int retOCLFFT = DoRECONSOperation(memObj_data[2], dataCUFFT_F, Newdim, numSlice, context, commandQueue, program, timesRecons, numThread);
	if(retOCLFFT == 1)
	{
		cout << "OCL FFT Operation is FAILED" << endl;
		return 1;
	}
	clReleaseMemObject(memObj_data[2]);
	GPU_end1 = clock();
	float GPUTimer_fft = timesRecons[0]+timesRecons[1]+timesRecons[2]+timesRecons[3];;
	GPU_start1  = clock();
	int xdim_old = dim[0];
	int ydim_old = dim[1];
	float* rss = (float*)malloc(nBytes_FRSS);
	int retRSS = DoRSSOCLOperation(memObj_data[1], rss, dim, sizeCfl, slice, numSlice, context, commandQueue, program, numThread);
	if(retRSS == 1)
	{
		cout << "RSS Operation is FAILED" << endl;
		return 1;
	}
	clReleaseMemObject(memObj_data[1]);
	GPU_end1 = clock();
	float GPUTimer_RSSRaw = (float)(GPU_end1 - GPU_start1)/CLOCKS_PER_SEC;
	float GPUTimer = GPUTimer_getspec + GPUTimer_zeroFill+ GPUTimer_fft + GPUTimer_RSSRaw;
	cout << "<path/filename> <Start Slice> <Number Slice Per Operatio> <New X dimension (Zero Filled Dimension)> <Number Thread (32 or 512>" << endl;
	cout << "<kspace> " << std::to_string(slice) << " " << std::to_string(numSlice) << " " << std::to_string(new_x) << " " << std::to_string(numThread)<< endl;
	cout << " " << endl;
	cout << "Timing - Get Specific Slice Data on GPU Processor : " << std::to_string(GPUTimer_getspec*1000) << " ms" << endl;
	cout << "Timing - Zero Filled - Interpolation on GPU Processor : " << std::to_string(GPUTimer_zeroFill*1000) << " ms" << endl;
	cout << "Timing - FFT + FFTSHIFT on CPU Processor : " << std::to_string(GPUTimer_fft*1000) << " ms" << endl;
	cout << "         Timing - FFT on GPU Processor : " << std::to_string(timesRecons[0]*1000) << " ms" << endl;
	cout << "         Timing - FFTSHIFT on GPU Processor : " << std::to_string(timesRecons[1]*1000) << " ms" << endl;
	cout << "         Timing - RSS on GPU Processor : " << std::to_string(timesRecons[2]*1000) << " ms" << endl;
	cout << "         Timing - Data Scaling on GPU Processor : " << std::to_string(timesRecons[3]*1000) << " ms" << endl;
	cout << "Timing - RSS Operation on GPU Processor : " << std::to_string(GPUTimer_RSSRaw*1000) << " ms" << endl;
	cout << "OPENCL GPU Timing using CPU Timer : " << std::to_string(GPUTimer*1000) << " ms" << endl;
	cout << "==========================================================================================================" << endl;
	/*int retCV = DoDisplayImage2CV(rss, dataCUFFT_F, xdim_old, ydim_old, slice, xdim_new, filename, numSlice);
	if(retCV == 1)
	{
		cout << "Display Image is Failed" << endl;
		return 1;
	}*/
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
	clReleaseDevice(device);
	free(filename);
	free(dim);
	free(rss);
	free(dataCUFFT_F);
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
int readCflFile(char *filen, float2* data)
{
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
int OCLgetSpecificSliceData(cl_mem data, cl_mem out, int* dim,  int spec_slice, int numSlice, int xdim_new, int ydim_new, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int slicedim = dim[2];
	int err = 0;
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoGetSpecificSliceDataOCL", &err);
	checkSuccess(err);
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &xdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(int), &ydim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 4, sizeof(int), &bitdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 5, sizeof(int), &slicedim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 6, sizeof(int), &numSlice));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 7, sizeof(int), &spec_slice));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to Set Kernel Argument(s)" << __FILE__ << ":" << __LINE__ << endl;
		exit(1);
	}
	int maxWI = numThread;
	int workItem_x = xdim;
	int workItem_y = ydim;
	int workGroup_x = 1;
	int workGroup_y = 1;
	if(xdim > maxWI)
	{
		workItem_x = maxWI;
		workItem_y = maxWI;
		workGroup_x = ceil(double(xdim)/double(workItem_x));
		workGroup_y = ceil(double(ydim)/double(workItem_y));
	}
	size_t workItemSize[2] = {workItem_x, workItem_y};
	size_t workGlobalSize[2] = {workGroup_x*workItem_x, workGroup_y*workItem_y};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, workGlobalSize, 0, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for kernel execution completion" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(err))
	{
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL Get Specific Slices Data" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
}
int OCLZeroFilled(cl_mem d_inputOneSlice, cl_mem d_inputOneSliceNew, int xdim, int ydim, int bitdim, int numSlice, int xdim_new, int ydim_new, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err = 0;
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoZeroFilledOCL", &err);
	checkSuccess(err);
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputOneSlice));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_inputOneSliceNew));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &xdim_new));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(int), &xdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 4, sizeof(int), &ydim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 5, sizeof(int), &bitdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 6, sizeof(int), &numSlice));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to Set Kernel Argument(s)" << __FILE__ << ":" << __LINE__ << endl;
		exit(1);
	}
	int maxWI = numThread;
	int workItem_x = xdim;
	int workItem_y = ydim;
	int workGroup_x = 1;
	int workGroup_y = 1;
	if(xdim > maxWI)
	{
		workItem_x = maxWI;
		workItem_y = maxWI;
		workGroup_x = ceil(double(xdim)/double(workItem_x));
		workGroup_y = ceil(double(ydim)/double(workItem_y));
	}
	size_t workItemSize[2] = {workItem_x, workItem_y};
	size_t workGlobalSize[2] = {workGroup_x*workItem_x, workGroup_y*workItem_y};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, workGlobalSize, 0, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for kernel execution completion" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(err))
	{
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL ZERO FILLED" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
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

int DoRECONSOperation( cl_mem inputData, float* output, int *dim, int numSlice, cl_context context, cl_command_queue commandQueue, cl_program program, float* times, int numThread)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int err = 0;
	clock_t start, end;
	int bitdim = dim[3];
	int sizeOneImage = xdim*ydim;
	int sizeManySlices = sizeOneImage * numSlice;
	size_t nBytes_F = sizeof(float)*sizeOneImage*numSlice;
	cl_mem d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes_F, NULL, &err);
	checkSuccess(err);
	start = clock();
	err = OCL_clfft(inputData, xdim, ydim, bitdim, numSlice, context, commandQueue);
	if(err != 0)
	{
		cerr << "Failed to do clFFT" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	end = clock();
	times[0] = (float)(end-start)/CLOCKS_PER_SEC;
	start = clock();
	err = OCL_fftshift(inputData, xdim, ydim, bitdim, numSlice, commandQueue, program,numThread);
	if(err != 0)
	{
		cerr << "Failed to do OCL FFT Shift" << __LINE__ << endl;
		return 1;
	}
	end = clock();
	times[1] = (float)(end-start)/CLOCKS_PER_SEC;
	start = clock();
	err = OCL_RSS(inputData, d_out, sizeOneImage, xdim, ydim, bitdim, numSlice, commandQueue, program, numThread);
	if(err != 0)
	{
		cerr << "Failed to do RSS Operation" << __LINE__ << endl;
		return 1;
	}
	end = clock();
	times[2] = (float)(end-start)/CLOCKS_PER_SEC;
	start = clock();
	err = OCL_dataScaling(d_out, sizeManySlices, context, commandQueue, program, numThread);	
	if(err != 0)
	{
		cerr << "Failed to do Data Scaling Operation" << __LINE__ << endl;
		return 1;
	}
	err = clEnqueueReadBuffer(commandQueue, d_out, CL_TRUE, 0, nBytes_F, output, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	clReleaseMemObject(d_out);
	end = clock();
	times[3] = (float)(end-start)/CLOCKS_PER_SEC;
	return 0;
}
int OCL_clfft(cl_mem inputData, int xdim, int ydim, int bitdim, int numSlice, cl_context context, cl_command_queue commandQueue)
{
	int err;
	int N = xdim * ydim;
	size_t offset = 0;
	size_t nBytes = sizeof(float2)*N;
	cl_mem temp = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes, NULL, &err);
	checkSuccess(err);
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[RANK] = {N};
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;

		return 1;
	}
	err = clfftCreateDefaultPlan(&planHandle, context, dim, clLengths);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
	// err = clfftSetPlanScale( planHandle, CLFFT_BACKWARD, 1 );
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
	// err = clfftSetPlanBatchSize(planHandle, batch);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	err = clfftBakePlan(planHandle, 1, &commandQueue, NULL, NULL);
	for(int i = 0; i < (bitdim*numSlice); i++)
	{
		offset = nBytes*i;
		err = clEnqueueCopyBuffer(commandQueue, inputData, temp, offset, 0, nBytes, 0, NULL, NULL);
		err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &commandQueue, 0, NULL, NULL, &temp, NULL, NULL);
		err = clFinish(commandQueue);
		err = clEnqueueCopyBuffer(commandQueue, temp, inputData, 0, offset, nBytes, 0, NULL, NULL);
	}
	err = clFinish(commandQueue);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	err = clfftDestroyPlan(&planHandle);
	checkSuccess(err);
	clfftTeardown();
	clReleaseMemObject(temp);
	return 0;
}
int OCL_fftshift(cl_mem inputData, int xdim, int ydim, int bitdim, int numSlice, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err;
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoOCLFFTShift", &err);
	if(err != 0)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(int), &xdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &bitdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(int), &numSlice));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to set Kernel Argument(s)" << __LINE__ << endl;
		return 1;
	}
	int maxWI = 32;
	int workItem_x = xdim;
	int workItem_y = ydim;
	int workGroup_x = 1;
	int workGroup_y = 1;
	if(xdim > maxWI)
	{
		workItem_x = maxWI;
		workItem_y = maxWI;
		workGroup_x = ceil(double(xdim)/double(workItem_x));
		workGroup_y = ceil(double(ydim)/double(workItem_y));
	}
	size_t workItemSize[2] = {workItem_x, workItem_y};
	size_t workGlobalSize[2] = {workGroup_x*workItem_x, workGroup_y*workItem_y};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, workGlobalSize, 0, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for kernel execution completion" << __LINE__ << endl;
		return 1;
	}
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL FFT SHIFT" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
}
int OCL_RSS(cl_mem inputData, cl_mem outData, int sizeOneImage, int xdim, int ydim, int bitdim, int numSlice, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err;
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoOCLRSS", &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outData));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &sizeOneImage));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(int), &xdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 4, sizeof(int), &ydim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 5, sizeof(int), &bitdim));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 6, sizeof(int), &numSlice));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to set Kernel Argument(s)" << __LINE__ << endl;
		return 1;
	}
	int maxWI = numThread;
	int workItem_x = sizeOneImage;
	int workGroup_x = 1;
	if(sizeOneImage > maxWI)
	{
		workItem_x = maxWI;
		workGroup_x = ceil(double(sizeOneImage)/double(workItem_x));
	}
	size_t workItemSize[1] = {workItem_x};
	size_t workGlobalSize[1] = {workGroup_x*workItem_x};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, workGlobalSize, workItemSize, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for Kernel Execution Completion" << __LINE__ << endl;
		return 1;
	}
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL RSS RECONS" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
}
int OCL_dataScaling(cl_mem d_out, int sizeOneImage, cl_context context, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err = 0;
	cl_mem max_buf;
	max_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
	err = OCLMax_reduce(d_out, max_buf, sizeOneImage, commandQueue, program, numThread);
	if(err != 0)
	{
		cerr << "Failed to do Max Reduce" << __LINE__ << endl;
		return 1;
	}
	err = OCLParallel_devide(d_out, max_buf, sizeOneImage, commandQueue, program, numThread);
	if(err != 0)
	{
		cerr << "Failed to do Parallel Devide" << __LINE__ << endl;
		return 1;
	}
	clReleaseMemObject(max_buf);
	return 0;
}
int OCLMax_reduce(cl_mem d_out, cl_mem max_buf, int sizeOneImage, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err = 0;
	size_t localMemSize = (4096*sizeof(float));
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoOCLMax_reduce", &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_out));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &max_buf));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &sizeOneImage));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 3, localMemSize, NULL));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to set Kernel Argument(s)" << __LINE__ << endl;
		return 1;
	}
	int maxWI = numThread;
	int workItem_x = sizeOneImage;
	int workGroup_x = 1;
	if(sizeOneImage > maxWI)
	{
		workItem_x = maxWI;
		workGroup_x = ceil(double(sizeOneImage)/double(workItem_x));
	}
	size_t workItemSize[1] = {workItem_x};
	size_t workGlobalSize[1] = {workGroup_x*workItem_x};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, workGlobalSize, workItemSize, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for Kernel Execution Completion" << __LINE__ << endl;
		return 1;
	}
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL MAX REDUCE" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
}
int OCLParallel_devide(cl_mem d_out, cl_mem max_buf, int sizeOneImage, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int err = 0;
	cl_kernel kernel;
	kernel = clCreateKernel(program, "DoOCLParallel_devide", &err);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	bool isKernelArgSuccess = true;
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_out));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &max_buf));
	isKernelArgSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), &sizeOneImage));
	if(!isKernelArgSuccess)
	{
		cerr << "Failed to set Kernel Argument(s)" << __LINE__ << endl;
		return 1;
	}
	int maxWI = numThread;
	int workItem_x = sizeOneImage;
	int workGroup_x = 1;
	if(sizeOneImage > maxWI)
	{
		workItem_x = maxWI;
		workGroup_x = ceil(double(sizeOneImage)/double(workItem_x));
	}
	size_t workItemSize[1] = {workItem_x};
	size_t workGlobalSize[1] = {workGroup_x*workItem_x};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, workGlobalSize, workItemSize, 0, NULL, NULL);
	if(!checkSuccess(clFinish(commandQueue)))
	{
		cerr << "Failed waiting for Kernel Execution Completion" << __LINE__ << endl;
		return 1;
	}
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}
	if(!checkSuccess(clReleaseKernel(kernel)))
	{
		cerr << "Failed to Release Kernel OCL PARALLEL DEVIDE" << __FILE__ << ":" << __LINE__ << endl;
		return 1;
	}
	return 0;
}
int DoRSSOCLOperation(cl_mem inputData, float* output, const int* dim, int size, int spec_slice, int numSlice, cl_context  context, cl_command_queue commandQueue, cl_program program, int numThread)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int err = 0;
	int bitdim = dim[3];
	int sizeOneImage = xdim*ydim;
	size_t nBytes_F = sizeof(float)*sizeOneImage*numSlice;
	cl_mem d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes_F, NULL, &err);
	checkSuccess(err);
	err = OCL_RSS(inputData, d_out, sizeOneImage, xdim, ydim, bitdim, numSlice, commandQueue, program, numThread);
	if(err != 0)
	{
		cerr << "Failed to do RSS Operation" << __LINE__ << endl;
		return 1;
	}
	err = clEnqueueReadBuffer(commandQueue, d_out, CL_TRUE, 0, nBytes_F, output, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << __LINE__ << endl;
		return 1;
	}	
	clReleaseMemObject(d_out);
	return 0;
}
cl_device_id create_device()
{
	cl_platform_id platform;
	cl_device_id device;
	int err;
	char platform_name[128];
    char device_name[128];
	err = oclGetPlatformID(&platform);
	size_t ret_param_size = 0;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME,
            sizeof(platform_name), platform_name,
            &ret_param_size);
	if(err < 0)
	{
		perror("Couldn't identify a Platform");
		exit(1);
	}
	else
	{
		printf("CL_PLATFORM_NAME : \t%s\n", platform_name );
	}
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	err = clGetDeviceInfo(device, CL_DEVICE_NAME,
            sizeof(device_name), device_name,
            &ret_param_size);
    printf("Device found on the above platform: %s\n", device_name);
	if(err < 0)
	{
		cerr << getErrorString(err) << __FILE__ << ":" << __LINE__ << endl;
		exit(1);
	}
	return device;
}
bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program)
{
    cl_int errorNumber = 0;
    ifstream kernelFile(filename.c_str(), ios::in);

    if(!kernelFile.is_open())
    {
        cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    ostringstream outputStringStream;
    outputStringStream << kernelFile.rdbuf();
    string srcStdStr = outputStringStream.str();
    const char* charSource = srcStdStr.c_str();
    *program = clCreateProgramWithSource(context, 1, &charSource, NULL, &errorNumber);
    if (!checkSuccess(errorNumber) || program == NULL)
    {
        cerr << "Failed to create OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    bool buildSuccess = checkSuccess(clBuildProgram(*program, 0, NULL, NULL, NULL, NULL));
    size_t logSize = 0;
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    if (logSize > 1)
    {
        char* log = new char[logSize];
        clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        string* stringChars = new string(log, logSize);
        delete[] log;
        delete stringChars;
    }
    if (!buildSuccess)
    {
        clReleaseProgram(*program);
        cerr << "Failed to build OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    return true;
}
bool checkSuccess(cl_int errorNumber)
{
    if (errorNumber != CL_SUCCESS)
    {
        cerr << "OpenCL error: " << getErrorString(errorNumber) << endl;
        return false;
    }
    return true;
}
const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        return -1000;
    }
    else 
    {
        if(num_platforms == 0)
        {
            printf("No OpenCL platform found!\n\n");
            return -2000;
        }
        else 
        {
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                printf("Failed to allocate memory for cl_platform ID's!\n\n");
                return -3000;
            }
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }
            if(*clSelectedPlatformID == NULL)
            {
                printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                *clSelectedPlatformID = clPlatformIDs[0];
            }
            free(clPlatformIDs);
        }
    }
    return CL_SUCCESS;
}