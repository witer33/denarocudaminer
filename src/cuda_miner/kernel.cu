#include <stdio.h>
#include <cuda_runtime.h>
#include "sha256.cuh"
#include <string.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

__device__ void bytes_slice(const unsigned char* str, unsigned char* buffer, size_t start, size_t end) {
	size_t j = 0;
	for (size_t i = start; i <= end; ++i) {
		buffer[j++] = str[i];
	}
	buffer[j] = 0;
}

__device__ bool my_strcmp(const char* str_a, const unsigned char* str_b, unsigned len) {
	for (int i = 0; i < len; i++) {
		if (str_a[i] != str_b[i])
			return false;
	}
	return true;
}

__device__ void sha256_to_hex(const unsigned char* data, char pout[64])
{

	const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
	int i = 0;
	for (; i < 32; ++i) {
		pout[i * 2] = hex[(data[i] >> 4) & 0xF];
		pout[i * 2 + 1] = hex[(data[i]) & 0xF];
	}
}

__device__ bool bytes_contains(const unsigned char* str_1, size_t str_1_len, unsigned char str_2) {
	for (int i = 0; i < str_1_len; i++) {
		if (str_1[i] == str_2)
			return true;
	}
	return false;
}

__device__ void compute_hash(void* input, size_t input_size, unsigned char* out) {
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char*)input, input_size);
	sha256_final(&ctx, out);
}

__device__ void lprint(const char* data, size_t length) {
	for (int x = 0; x < length; x++)
		printf("%c", data[x]);
	printf("\n");
}

__global__ void miner_thread(const unsigned char* hash_prefix, const unsigned char* last_block_chunk, size_t hash_prefix_length, size_t difficulty, const unsigned char* charset, size_t charset_len, int *stop, int step) {
	
	// Starting point of i.
	uint64_t start = blockIdx.x * blockDim.x + threadIdx.x;

	// Distribution algorithm (5 threads example):
	// 0 1 2 3 4 (starting points)
	// 5 6 7 8 9 (adding the step which is the number of threads)
	// 10 11 12 13 14 ...

	// Allocating local variables.
	size_t temp_size = hash_prefix_length + 4;
	uint32_t i = start;
	unsigned char temp[300];
	
	// Copying hash prefix to temp
	memcpy(temp, hash_prefix, hash_prefix_length);
	unsigned char* nonce_pointer = temp + hash_prefix_length;

	unsigned char out[32];
	char hash_hex[64];


	while (*stop == 0) {
		// Adding hash random to temp
		memcpy(nonce_pointer, &i, 4);

		// Computing hash
		compute_hash(temp, temp_size, out);

		// Turning it to hex
		sha256_to_hex(out, hash_hex);

		// Checking if it's valid
		if (my_strcmp(hash_hex, last_block_chunk, difficulty) && bytes_contains(charset, charset_len, hash_hex[difficulty])) {
			// If it's valid stop all threads and print the random.
			printf("%d\n", i);
			*stop = -1;
			break;
		} else if (i == 4294967295) {
			// If it reaches the uint32 maximum print 0 and stop all threads.
			*stop = -1;
			printf("0\n");
			break;
		}
		// Add the step to i (blocks * threads)
		i += step;
	}
}

int char2int(char input) {
	if (input >= '0' && input <= '9')
		return input - '0';
	if (input >= 'A' && input <= 'F')
		return input - 'A' + 10;
	if (input >= 'a' && input <= 'f')
		return input - 'a' + 10;
	throw std::invalid_argument("Invalid input string");
}

void hex2bin(const char* src, char* target) {
	while (*src && src[1])
	{
		*(target++) = char2int(*src) * 16 + char2int(src[1]);
		src += 2;
	}
}

void hex_print(const unsigned char* data, size_t length) {
	for (int x = 0; x < length; x++)
		printf("%02X", data[x]);
	printf("\n");
}

int main(int argc, char** argv) {

	if (argc < 5)
		return -1;

	// Settings
	int blocks = 50;
	int threads = 512;

	// Console arguments (last_block_chunk charset hex_hash_prefix difficulty)
	int difficulty = std::stoi(argv[4]);
	std::string _last_block_chunk(argv[1]);
	std::string _charset(argv[2]);
	std::string hex_hash_prefix(argv[3]);
	size_t hash_prefix_length = (size_t) (hex_hash_prefix.length() / 2);

	// Hex hash prefix to bytes hash prefix
	unsigned char* hash_prefix;
	char* temp_hash_prefix;
	temp_hash_prefix = static_cast<char*>(malloc(hash_prefix_length));
	cudaMallocManaged(&hash_prefix, hash_prefix_length);
	hex2bin(hex_hash_prefix.c_str(), temp_hash_prefix);
	cudaMemcpy(hash_prefix, (unsigned char *) temp_hash_prefix, hash_prefix_length, cudaMemcpyHostToDevice);
	free(temp_hash_prefix);

	// Allocating global memory variables
	unsigned char* last_block_chunk;
	unsigned char* charset;
	int* stop;
	size_t charset_length = _charset.length();

	pre_sha256();
	cudaMallocManaged(&stop, sizeof(int));
	cudaMallocManaged(&charset, charset_length);
	cudaMallocManaged(&last_block_chunk, difficulty);
	
	// Copying to global memory variables
	cudaMemcpy(charset, _charset.c_str(), charset_length, cudaMemcpyHostToDevice);
	cudaMemcpy(last_block_chunk, _last_block_chunk.c_str(), difficulty, cudaMemcpyHostToDevice);
	*stop = 0;

	int step = blocks * threads;
	// GPU starts

	// Starting threads.
	miner_thread <<<blocks, threads>>> (hash_prefix, last_block_chunk, hash_prefix_length, (size_t)difficulty, charset, charset_length, stop, step);

	// Waiting for completion, and verifying if there was any error.
	checkCudaErrors(cudaDeviceSynchronize());

	//GPU ends


	// Freeing global variables...
	cudaFree(hash_prefix);
	cudaFree(stop);
	cudaFree(charset);
	cudaFree(last_block_chunk);

	return 0;
}
