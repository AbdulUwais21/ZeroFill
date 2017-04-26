FILE 		= Trial3

INCLUDE 	= 	-I./ \
				-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" \
				-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include\helper" \
				-I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" \
				-I"E:\Newfolder\Compilers\opencv2413\opencv\build\include" \
				-I"E:\Newfolder\LIBRARIES\FFTW\64" \
				-I"E:\Live\Eks\PLAY_CUDA\OPENCL\People_CODE\NVIDIA GPU Computing SDK\shared\inc" \
				-I"E:\Live\Eks\PLAY_CUDA\OPENCL\People_CODE\clFFT-master\src\include"

LIB 		= -L./ -l libCommon -l clFFT -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64" -l OpenCL\
				-L"E:\Newfolder\Compilers\opencv2413\opencv\build\x64\vc12\lib" -lopencv_core2413 -lopencv_highgui2413 -lopencv_imgproc2413 \
				
CU_FLAG 	=   
UNUSED		=	--ptxas-options=-v -lineinfo -Xcompiler -rdynamic -O2 -default-stream per-thread -arch=sm_50 -cudart static -O2 -Xptxas -v -default-stream per-thread \
				-L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64" -l cuda -l cudart -l cudart_static -l cufft \
				-L"E:\Newfolder\LIBRARIES\FFTW\64" -llibfftw3f-3
PRINTPTX	= 	-ptx -src-in-ptx
				

.PHONY: clean

%: %.cu
	nvcc $(INCLUDE) $(LIB) $(CU_FLAG) -o $@ $<
txt: $(FILE).exe
	./$(FILE) > output.txt
clean: 
	rm -f *.exe
