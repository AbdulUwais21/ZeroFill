//; d_inputOneSlice, d_inputOneSliceNew, xdim_new, xdim, ydim, bitdim)
__kernel void DoZeroFilledOCL(__global float2* input, __global float2* out, const int xdim_new, const int xdim, const int ydim, const int bitdim)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	int indexOne = idy * xdim + idx;
	int N = xdim*ydim;
	int residue = (xdim_new-xdim)/2;
	int sidx = (xdim_new*residue)+residue;
	int offset = 2*residue;
	int offsetPerBitmask = xdim_new*xdim_new;
	if(idx <= xdim && idy <= ydim)
	{
		for(int m = 0; m < bitdim; m++)
		{
			for(int l = 0; l < numSlice; l++)
			{
				
			}
			out[sidx + indexOne + (idy*offset)+ (m*offsetPerBitmask)].x = input[indexOne + (m*N)].x;
			out[sidx + indexOne + (idy*offset)+ (m*offsetPerBitmask)].y = input[indexOne + (m*N)].y;
		}
	}


}

__kernel void DoOCLFFTShift(__global float2* data, const int N, const int bit)
{
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;

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
        		regTemp = data[index + (i*sSlice)]; // I am add "+ (i*sSlice)"

        		// First Quad
        		data[index+ (i*sSlice)] = data[index + sEq1 + (i*sSlice)]; // I am add "+ (i*sSlice)"

        		// Third Quad
        		data[index + sEq1+ (i*sSlice)] = regTemp; // I am add "+ (i*sSlice)"
        	}
           
        }
    }
    else
    {
        if (yIndex < N / 2)
        {
        	for(int i = 0; i< bit; i++)
        	{
        		regTemp = data[index + (i*sSlice)]; // I am add "+ (i*sSlice)"

        		// Second Quad
        		data[index + (i*sSlice)] = data[index + sEq2 + (i*sSlice)]; // I am add "+ (i*sSlice)"

        		// Fourth Quad
        		data[index + sEq2 + (i*sSlice)] = regTemp; // I am add "+ (i*sSlice)"
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

__kernel void DoOCLRSS(__global float2* input, __global float* out, const int N, const int x, const int y, const int bit)
{
	unsigned int tidx = get_global_id(0);
	// int sizeOneBit = x * y;
	float temp = 0;
	if(tidx < N) 
	{
		for(int i = 0; i < bit; i++)
		{
			// input[(i*sizeOneBit)+tidx].x = (float)pow((double)input[(i*sizeOneBit)+tidx].x, (double)2); //Try to using bitwise operator. It is faster than pow 2
			// input[(i*sizeOneBit)+tidx].y = (float)pow((double)input[(i*sizeOneBit)+tidx].y, (double)2); //Try to using bitwise operator. It is faster than pow 2
			// temp = (float)sqrt((pow((double)input[(i*sizeOneBit)+tidx].x, (double)2))+(pow((double)input[(i*sizeOneBit)+tidx].y, (double)2)));

			temp = clCabsf(input[(i*N)+tidx]);
			temp = pow((double)temp, (double)2);
			temp += temp;
		}
		temp = (float)sqrt(temp);
		out[tidx] = temp;
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