#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifndef WRITE_OUTPUT
#define WRITE_OUTPUT 0   // set to 1 to write merged output via MPI-IO
#endif

#ifndef BLOCK_ELEMS
#define BLOCK_ELEMS 1000000LL   // buffer size per rank (like SEGMENT_SIZE)
#endif

// Our implicit sorted arrays:
// A[i] = 2*i, B[j] = 2*j + 1
static inline long long getA(long long i) { return (2LL * i); }
static inline long long getB(long long j) { return (2LL * j + 1LL); }

// Find partition i such that i + j = k and A[i-1] <= B[j] and B[j-1] < A[i]
// (standard merge-path partition)
static long long find_partition_implicit(long long n, long long k) {
    // i in [max(0, k-n), min(n, k)]
    long long low  = (k > n) ? (k - n) : 0;
    long long high = (k < n) ? k : n;

    while (low < high) {
        long long i = low + (high - low) / 2;
        long long j = k - i;

        // Need A[i] >= B[j-1] to stop moving right
        // Move right if i < n and j > 0 and A[i] < B[j-1]
        if (i < n && j > 0 && (long long)getA(i) < (long long)getB(j - 1)) {
            low = i + 1;
        } else {
            high = i;
        }
    }
    return low;
}

// Stream-merge exactly out_len items starting from A[a0],B[b0] into buffer.
// Returns updated indices via pointers.
static void stream_merge_to_buffer(long long n,
                                   long long *a_idx, long long *b_idx,
                                   long long *buf, long long out_len) {
    long long i = *a_idx;
    long long j = *b_idx;

    for (long long t = 0; t < out_len; t++) {
        long long av = (i < n) ? getA(i) : LLONG_MAX;
        long long bv = (j < n) ? getB(j) : LLONG_MAX;
        if (av <= bv) { buf[t] = av; i++; }
        else          { buf[t] = bv; j++; }
    }

    *a_idx = i;
    *b_idx = j;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: srun -n <p> ./parallel_merge_streaming <n_per_array>\n");
            printf("Example: srun -n 8 ./parallel_merge_streaming 10000000\n");
        }
        MPI_Finalize();
        return 1;
    }

    long long n = atoll(argv[1]);     // elements in each array
    long long total = 2LL * n;        // total output length

    // Each rank owns an output range [s,e)
    long long base = total / p;
    long long rem  = total % p;
    long long s = (long long)rank * base + (rank < rem ? rank : rem);
    long long e = s + base + (rank < rem ? 1 : 0);
    long long my_out = e - s;

    // Compute merge-path partitions for start/end of our output slice
    long long a_s = find_partition_implicit(n, s);
    long long a_e = find_partition_implicit(n, e);
    long long b_s = s - a_s;
    long long b_e = e - a_e;

    // Sanity: our local merge will consume (a_e-a_s) from A and (b_e-b_s) from B
    // and produce my_out items.
    if ((a_e - a_s) + (b_e - b_s) != my_out) {
        if (rank == 0) fprintf(stderr, "Partition mismatch!\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

#if WRITE_OUTPUT
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "merged.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
#endif

    long long *buf = (long long *)malloc((size_t)BLOCK_ELEMS * sizeof(long long));
    if (!buf) {
        fprintf(stderr, "Rank %d: buffer malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    long long a = a_s, b = b_s;
    long long produced = 0;

    // For correctness checking without gathering:
    long long first_val = 0, last_val = 0;
    int have_any = 0;

    while (produced < my_out) {
        long long chunk = my_out - produced;
        if (chunk > BLOCK_ELEMS) chunk = BLOCK_ELEMS;

        stream_merge_to_buffer(n, &a, &b, buf, chunk);

        if (!have_any && chunk > 0) {
            first_val = buf[0];
            have_any = 1;
        }
        if (chunk > 0) last_val = buf[chunk - 1];

#if WRITE_OUTPUT
        // MPI_File_write_at_all takes int count, so we chunk to stay safe.
        MPI_Offset off_bytes = (MPI_Offset)((s + produced) * (long long)sizeof(int));
        // chunk <= BLOCK_ELEMS fits in int for BLOCK_ELEMS <= 2e9
        MPI_File_write_at_all(fh, off_bytes, buf, (int)chunk, MPI_INT, MPI_STATUS_IGNORE);
#endif

        produced += chunk;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Neighbor boundary check (no root gather):
    // Ensure last(rank-1) <= first(rank)
    long long prev_last = 0;
    if (rank > 0) {
        MPI_Recv(&prev_last, 1, MPI_LONG_LONG, rank - 1, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < p - 1) {
        MPI_Send(&last_val, 1, MPI_LONG_LONG, rank + 1, 111, MPI_COMM_WORLD);
    }

    int ok_local = 1;
    if (rank > 0 && have_any && prev_last > first_val) ok_local = 0;

    int ok_global = 0;
    MPI_Reduce(&ok_local, &ok_global, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("n per array: %lld (total %lld), ranks=%d\n", n, total, p);
        printf("Streaming merge time (no scatter/gather): %.6f s\n", t1 - t0);
        printf("Boundary-sorted check: %s\n", ok_global ? "PASS" : "FAIL");
#if WRITE_OUTPUT
        printf("Output written to merged.bin (binary ints)\n");
#endif
    }

#if WRITE_OUTPUT
    MPI_File_close(&fh);
#endif
    free(buf);

    MPI_Finalize();
    return 0;
}
