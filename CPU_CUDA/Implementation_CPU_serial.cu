#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <cufft.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fftw3.h>
typedef float2 cplx;
#define RANK 1
#define M_PI 3.14159265358979323846
#define fftshift(out, in, x, y, bitdim, numSlice) circshift(out, in, x, y, bitdim, (x/2), (y/2), numSlice)
void readHeaderFile(char *filen, int * dims);
int readCflFile(char *file, cplx* data);
void getSpecificSliceData(cplx* input, cplx* out, cplx* rawOut, int spec_slice, int spec_bit, const int *dim, int x_new, int y_new, float* times);
void dataScaling(float* data, int size);
int DoDisplayImage2CV(float* dataRaw, float* dataRecons, int xdim, int ydim, int spec_slice, int zipxdim, int numSlice);
void circshift(cplx *out, cplx *in, int xdim, int ydim, int bitdim, int xshift, int yshift, int numSlice);
void DoRSSCPU(cplx* input, float* out, int N, int x, int y, int bitdim, int numSlice);
int DoRECONSOperation( cplx* DatafftOneSlice, float* out, int *dim, int numSlice, float* times);
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
	if(argc < 5)
	{
		cout << "<Usage> <path/filename> <Start Slice> <Number Slice Per Operatio> <New X dimension (Zero Filled Dimension)>" << endl;
		return 1;
	}
	clock_t CPU_start1, CPU_end1;
	char *filename = (char*)malloc(100);
	int *dim = (int*)malloc(sizeof(int)*4);
	int *Newdim = (int*)malloc(sizeof(int)*4);
	float *timesRecons = (float*)malloc(sizeof(float)*4);
	float *timesRaw = (float*)malloc(sizeof(float)*2);
	sprintf(filename,"%s",argv[1]);
	int slice = atoi(argv[2]);
	int numSlice = atoi(argv[3]);
	int new_x = atoi(argv[4]);
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
	cplx *data = (cplx*)malloc(sizeof(cplx)*sizeCfl);
	int ret = readCflFile(filename, data);
	if(ret == 1)
	{
		cout << "Error on Reading CFL File" << endl;
		return 1;
	}
	CPU_start1  = clock();
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int xdim_new = new_x;
	int ydim_new = xdim_new;
	int sizeImageNew = xdim_new*ydim_new * numSlice;
	int sizeImage = xdim*ydim*numSlice;
	int sizeImageNewSlice = sizeImageNew* bitdim;
	int sizeImageNewSliceRaw = xdim*ydim* bitdim * numSlice;
	size_t nBytes_C = sizeof(cplx)*sizeImageNewSlice;
	size_t nBytes_CRaw = sizeof(cplx)*sizeImageNewSliceRaw;
	size_t nBytes_F = sizeof(float)*sizeImageNew;
	size_t nBytes_FRSS = sizeof(float)*sizeImage;
	cplx* dataManySlice = (cplx*)malloc(nBytes_C);
	cplx* dataManySliceRaw = (cplx*)malloc(nBytes_CRaw);
	memset(dataManySlice, 0, nBytes_C);
	getSpecificSliceData(data, dataManySlice, dataManySliceRaw, slice, numSlice, dim, xdim_new, ydim_new, timesRaw);
	free(data);
	Newdim[0] = xdim_new;
	Newdim[1] = ydim_new;
	xdim = dim[0];
	ydim = dim[1];
	bitdim = dim[3];
	int sizeManyImage = xdim*ydim*numSlice;
	CPU_end1 = clock();
	float CPUTimer_getspec = timesRaw[0] + timesRaw[1];
	CPU_start1  = clock();
	float* dataFFT_F = (float*)malloc(nBytes_F);
	int retCUFFT = DoRECONSOperation(dataManySlice, dataFFT_F, Newdim, numSlice, timesRecons);
	if(retCUFFT == 1)
	{
		cout << "CPU RECONSTRUCTION Operation is FAILED" << endl;
		return 1;
	}
	CPU_end1 = clock();
	float CPUTimer_fft = timesRecons[0]+timesRecons[1]+timesRecons[2]+timesRecons[3];
	CPU_start1  = clock();
	float *rss = (float*)malloc(nBytes_FRSS);
	DoRSSCPU(dataManySliceRaw, rss, sizeManyImage, xdim, ydim, bitdim, numSlice);
	CPU_end1 = clock();
	float CPUTimer_RSS = (float)(CPU_end1 - CPU_start1)/CLOCKS_PER_SEC;
	float CPUTimer = CPUTimer_getspec + CPUTimer_fft + CPUTimer_RSS;
	cout << "==========================================================================================================" << endl;
	cout << "<path/filename> <Start Slice> <Number Slice Per Operatio> <New X dimension (Zero Filled Dimension)>" << endl;
	cout << "<kspace> " << std::to_string(slice) << " " << std::to_string(numSlice) << " " << std::to_string(new_x) << endl;
	cout << " " << endl;
	cout << "Timing - Get Specific Slice Data on CPU Processor : " << std::to_string(timesRaw[0]*1000) << " ms" << endl;
	cout << "Timing - Zero Filling Interpolation on CPU Processor : " << std::to_string(timesRaw[1]*1000) << " ms" << endl;
	cout << "Timing - RECONSTRUCTION on CPU Processor : " << std::to_string(CPUTimer_fft*1000) << " ms" << endl;
	cout << "         Timing - FFT on CPU Processor : " << std::to_string(timesRecons[0]*1000) << " ms" << endl;
	cout << "         Timing - FFTSHIFT on CPU Processor : " << std::to_string(timesRecons[1]*1000) << " ms" << endl;
	cout << "         Timing - RSS on CPU Processor : " << std::to_string(timesRecons[2]*1000) << " ms" << endl;
	cout << "         Timing - Data Scaling on CPU Processor : " << std::to_string(timesRecons[3]*1000) << " ms" << endl;
	cout << "Timing - RAW Operation on CPU Processor : " << std::to_string(CPUTimer_RSS*1000) << " ms" << endl;
	cout << "CPU Timing using CPU Timer : " << std::to_string(CPUTimer*1000) << " ms" << endl;
	cout << "==========================================================================================================" << endl;
	/*int retCV = DoDisplayImage2CV(rss, dataFFT_F, xdim, ydim, slice, xdim_new, numSlice);
	if(retCV == 1)
	{
		cout << "Display Image is Failed" << endl;
		return 1;
	}*/
	free(filename);
	free(dim);
	free(Newdim);
	free(timesRecons);
	free(timesRaw);
	free(rss);
	free(dataManySlice);
	free(dataManySliceRaw);
	free(dataFFT_F);
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
	streampos size;
	char path[20];
	sprintf(path,"%s.cfl",filen);
	ifstream file(path, ios::in | ios::binary | ios::ate);
	if(file.is_open())
	{
		size = file.tellg();
		// cout << "Contains Size : "<< std::to_string(size) << endl;
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
void getSpecificSliceData(cplx* input, cplx* out, cplx* rawOut, int spec_slice, int numSlice, const int* dim, int xdim_new, int ydim_new, float* times )
{
	clock_t start,end;
	int xdim = dim[0];
	int ydim = dim[1];
	int slicedim = dim[2];
	int bitdim = dim[3];
	int sizeOneImage = xdim*ydim;
	int residue = (xdim_new-xdim)/2;
	int sidx = (xdim_new*residue)+residue;
	int offset = 2*residue;
	int offsetPerBitmask = numSlice*xdim_new*ydim_new;
	int offsetPerSlice = xdim_new*ydim_new;
	int index = 0;
	start = clock();
	for(int m = 0; m < bitdim; m++)
	{
		for(int l = spec_slice; l < (spec_slice+numSlice); l++)
		{
			for(int k = 0; k < ydim; k++)
			{
				for(int j = 0; j < xdim; j++)
				{
					rawOut[index]  = input[j + (k*xdim) + (l*sizeOneImage) + (m*slicedim*sizeOneImage)];
					index++;
				}
			}
		}
	}
	end = clock();
	times[0] = (float)(end-start)/CLOCKS_PER_SEC;
	start = clock();
	for(int m = 0; m < bitdim; m++)
	{
		for(int l = 0; l < numSlice; l++)
		{
			for(int i = 0; i < ydim ; i++)
			{
				for(int j = 0; j < xdim; j++)
				{
					out[sidx + j + (i*xdim) + (i*offset) + (l*offsetPerSlice) +(m*offsetPerBitmask)] = rawOut[j + (i*xdim) + (l*sizeOneImage) + (m*numSlice*sizeOneImage)];
				}
			}
		}
	}
	end = clock();
	times[1] = (float)(end-start)/CLOCKS_PER_SEC;
}
void DoRSSCPU(cplx* input, float* out, int N, int x, int y, int bitdim, int numSlice)
{
	float *temp = (float*)malloc(sizeof(float)*N);
	for(int m = 0; m < bitdim; m++)
	{
		for(int i = 0; i < numSlice ; i++)
		{
			for(int j = 0; j < y; j++)
			{
				for(int k = 0; k < x; k++)
				{
					out[k + (j*x) + (i*y*x)] = cuCabsf(input[k + (j*x)+ (i*y*x) + (m*x*y*numSlice)]);
					out[k + (j*x) + (i*y*x)] = (float)pow((out[k + (j*x) + (i*y*x)]),2);
					temp[k + (j*x) +(i*y*x)] += out[k + (j*x) + (i*y*x)];
				}
			}
		}
	}
	for(int i = 0; i < numSlice ; i++)
	{
		for(int j = 0; j < y; j++)
		{
			for(int k = 0; k < x; k++)
			{
				out[k + (j*x)+ (i*y*x)] = 0;
				out[k + (j*x)+ (i*y*x)] = sqrt(temp[k + (j*x)+ (i*y*x)]);
			}
		}
	}
	free(temp);
}
int DoDisplayImage2CV(float* dataRaw, float* dataRecons, int xdim, int ydim, int spec_slice, int zipdimx, int numSlice)
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
		cv::waitKey(2000);
	}
	cv::waitKey();
	free(oneImage);
	free(oneImageRaw);
	return 0;
}
int DoRECONSOperation( cplx* DatafftManySlice, float* out, int *dim, int numSlice, float* times)
{
	int xdim = dim[0];
	int ydim = dim[1];
	int bitdim = dim[3];
	int sizeOneSlice = xdim*ydim*bitdim;
	int sizeManySlice = sizeOneSlice*numSlice;
	int sizeManyImage = xdim*ydim*numSlice;
	int sizeOneImage = xdim*ydim;
	size_t nBytes_C = sizeof(cplx)*sizeManySlice;
	cplx* temp_in = (cplx*)malloc(nBytes_C);
	clock_t start, end;
	start = clock();
	fftwf_plan pfftw;
	int n[] = {sizeOneImage};
	int howmany = bitdim*numSlice;
	int idist = sizeOneImage;
	int odist = sizeOneImage;
	int istride = 1;
	int ostride = 1;
	int *onembed = NULL;
	int *inembed = NULL;
	pfftw = fftwf_plan_many_dft(RANK, n, howmany, 
								reinterpret_cast<fftwf_complex*>(DatafftManySlice),
								inembed, istride, idist, 
								reinterpret_cast<fftwf_complex*>(DatafftManySlice),
								onembed, ostride, odist,
								FFTW_BACKWARD,
								FFTW_ESTIMATE);
	fftwf_execute(pfftw);
	fftwf_destroy_plan(pfftw);
	fftwf_cleanup();
	end = clock();
    times[0] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock(); 
	memcpy(temp_in, DatafftManySlice, nBytes_C);
	fftshift(DatafftManySlice, temp_in, xdim, ydim, bitdim, numSlice);
	free(temp_in);
	end = clock();
    times[1] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock();
	DoRSSCPU(DatafftManySlice, out, sizeManyImage, xdim, ydim, bitdim, numSlice);
	end = clock();
    times[2] = (float)(end-start)/CLOCKS_PER_SEC;
    start = clock();
	dataScaling(out, sizeManyImage);
	end = clock();
    times[3] = (float)(end-start)/CLOCKS_PER_SEC;
	return 0;
}
void circshift(cplx *out, cplx *in, int xdim, int ydim, int bitdim, int xshift, int yshift, int numSlice)
{
	int N = xdim*ydim;
	int Bitm = N*numSlice;
	for(int m = 0; m < bitdim; m++)
	{
		for(int l = 0; l < numSlice ; l++)
		{
		  for (int i =0; i < xdim; i++) {
		    int ii = (i + xshift) % xdim;
		    for (int j = 0; j < ydim; j++) {
		      int jj = (j + yshift) % ydim;
		      out[ii * ydim + jj + (l*N) + (m*Bitm)] = in[i * ydim + j + (l*N) + (m*Bitm)];
		    }
		  }
		}
	}
}