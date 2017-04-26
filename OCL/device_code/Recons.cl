//; d_inputOneSlice, d_inputOneSliceNew, xdim_new, xdim, ydim, bitdim)
__kernel void DoGetSpecificSliceDataOCL(__global float2* input, __global float2* out, const int xdim, const int ydim, const int bitdim, const int slicedim, const int numSlice, const int spec_slice)
{

	int idx = get_global_id(0);
	int idy = get_global_id(1);
	int indexOne = idy * xdim + idx;
	int N = xdim*ydim;
	// int residue = (xdim_new-xdim)/2;
	// int sidx = (xdim_new*residue)+residue;
	// int offset = 2*residue;
	// int offsetPerBitmask = xdim_new*xdim_new;
	// int offsetPerSlice = xdim_new*xdim_new;
	int offsetPerBitmask = slicedim*N;
	int offsetBitmask = N*numSlice;
	int offsetSlice = N;
	int sliceP = 0;
	if(idx <= xdim && idy <= ydim)
	{
		for(int m = 0; m < bitdim; m++)
		{
			sliceP = 0;
			for(int l = spec_slice; l < (spec_slice+numSlice); l++)
			{
				out[indexOne + (sliceP*offsetSlice) + (m*offsetBitmask)] = input[indexOne + (l*N) + (m*offsetPerBitmask)];
				// out[sidx + indexOne + (idy*offset) + (sliceP*offsetPerSlice) + (m*offsetPerBitmask)] = outRaw[indexOne + (sliceP*offsetSlice) + (m*offsetBitmask)];
				sliceP++;
			}
			
		}
	}
}

__kernel void DoZeroFilledOCL(__global float2* input, __global float2* out, const int xdim_new, const int xdim, const int ydim, const int bitdim, const int numSlice)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	int indexOne = idy * xdim + idx;
	int N = xdim*ydim;
	int NNew = xdim_new*xdim_new;
	int residue = (xdim_new-xdim)/2;
	int sidx = (xdim_new*residue)+residue;
	int offset = 2*residue;
	int offsetPerBitmaskNew = NNew*numSlice;
	int offsetPerSliceNew = NNew;
	int offsetPerBitmask = N*numSlice;
	int offsetPerSlice = N;

	if(idx <= xdim && idy <= ydim)
	{
		for(int m = 0; m < bitdim; m++)
		{
			for(int l = 0; l < numSlice; l++)
			{
				out[sidx + indexOne + (idy*offset)+ (l*offsetPerSliceNew) + (m*offsetPerBitmaskNew)] = input[indexOne + (l*offsetPerSlice)+(m*offsetPerBitmask)];
				//out[sidx + indexOne + (idy*offset)+ (l*offsetPerSliceNew) + (m*offsetPerBitmaskNew)].y = input[indexOne + (l*offsetPerSlice)+(m*offsetPerBitmask)].y;
			}
		}
	}
}

__kernel void DoOCLFFTShift(__global float2* data, const int N, const int bit, const int numSlice)
{
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;
    int manySlices = sSlice * numSlice;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    // Thread Index (1D)
    int xThreadIdx = get_local_id(0);
    int yThreadIdx = get_local_id(1);

    // Block Width & Height
    int blockWidth = get_num_groups(0);
    int blockHeight = get_num_groups(1);

    // Thread Index (2D)
    int xIndex = get_global_id(0);
    int yIndex = get_global_id(1);

    // Thread Index Converted into 1D Index
    int index = (yIndex * N) + xIndex;

    float2 regTemp;

    if (xIndex < N / 2)
    {
        if (yIndex < N / 2)
        {
        	for(int i = 0; i<bit; i++)
        	{
        		for(int j = 0; j < numSlice; j++)
        		{
        			regTemp = data[index + (j*sSlice) + (i * manySlices)]; // I am add "+ (i*sSlice)"

        			// First Quad
        			data[index+ (j*sSlice) + (i * manySlices)] = data[index + sEq1 + (j*sSlice) + (i * manySlices)]; // I am add "+ (i*sSlice)"

        			// Third Quad
        			data[index + sEq1+ (j*sSlice) + (i * manySlices)] = regTemp; // I am add "+ (i*sSlice)"
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
        			regTemp = data[index + (j*sSlice) + (i * manySlices)]; // I am add "+ (i*sSlice)"

        			// Second Quad
        			data[index + (j*sSlice) + (i * manySlices)] = data[index + sEq2 + (j*sSlice) + (i * manySlices)]; // I am add "+ (i*sSlice)"

        			// Fourth Quad
        			data[index + sEq2 + (j*sSlice) + (i * manySlices)] = regTemp; // I am add "+ (i*sSlice)"
        		}
        	}
            
        }
    }
}

float clCrealf (float2 x) 
{ 
    return x.x; 
}

float clCimagf (float2 x) 
{ 
    return x.y; 
}

float clCabsf (float2 x)
{
    float a = clCrealf(x);
    float b = clCimagf(x);
    float v, w, t;
    a = fabs(a);
    b = fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sqrt(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}

__kernel void DoOCLRSS(__global float2* input, __global float* out, const int N, const int x, const int y, const int bit, const int numSLice)
{
	unsigned int tidx = get_global_id(0);
	float temp = 0;
	float sum = 0;

	int sizeManySlices = N * numSLice;
	if(tidx < N) 
	{
		for(int j = 0; j < numSLice; j++)
		{
			temp = 0;
			sum = 0;
			for(int i = 0; i < bit; i++)
			{
				temp = clCabsf(input[tidx + (j*N)+ (i*sizeManySlices)]);
				temp = pow((double)temp, (double)2);
				sum += temp;
			}
			sum = (float)sqrt(sum);
			out[tidx + (j*N)] = sum;
		}
	}
}

// http://stackoverflow.com/questions/18950732/atomic-max-for-floats-in-opencl
// //Function to perform the atomic max
 inline void AtomicMaxf(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal,operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__constant float FLOAT_MAX = 1;

__kernel void DoOCLMax_reduce(__global float* d_array, __global float* d_max, 
                                              const int elements, __local float* shared)
{
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    shared[tid] = -FLOAT_MAX; 

    while (gid < elements) {
        shared[tid] = fmax(shared[tid], d_array[gid]);
        gid += get_num_groups(0);
        }
     barrier(CLK_LOCAL_MEM_FENCE);
    gid = get_global_id(0);  // 1
    for (unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        	barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
      AtomicMaxf(d_max, shared[0]);
}

__kernel void DoOCLParallel_devide(__global float * data, __global float* max, int size)
{
	int idx = get_global_id(0);

	if(idx < size)
	{
		data[idx] = data[idx]/max[0];
	}
}