#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>

// Platform-specific aligned memory handling
#if defined(_MSC_VER)
  #include <malloc.h>
  #define aligned_free _aligned_free
#else
  #define aligned_free free
#endif

// ----------- Tunables -----------
// Memory alignment for potential performance benefits (cache line alignment)
#define ALIGN_BYTES 64

// Control whether to gather final result on root process (expensive for large arrays)
#ifndef DO_GATHER_RESULT
#define DO_GATHER_RESULT 1
#endif

// Control whether to time only the merge operation or include gather
#ifndef TIME_MERGE_ONLY
#define TIME_MERGE_ONLY 1
#endif

/**
 * Allocate memory aligned to ALIGN_BYTES boundary
 * Aligned memory can improve cache performance
 */
static void *aligned_malloc(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, ALIGN_BYTES);
#else
    void *p = NULL;
    if (posix_memalign(&p, ALIGN_BYTES, bytes) != 0) return NULL;
    return p;
#endif
}

/**
 * Sequential merge of two sorted arrays
 * Two-pointer merge algorithm
 * 
 * @param a First sorted array
 * @param n1 Length of first array
 * @param b Second sorted array
 * @param n2 Length of second array
 * @param out Output array (must have space for n1+n2 elements)
 */
static inline void sequential_merge_ptr(const int *a, int n1, const int *b, int n2, int *out) {
    const int *pa = a, *pb = b;           // Pointers to current position in each array
    const int *ea = a + n1, *eb = b + n2; // End pointers
    int *po = out;                         // Output pointer

    // Merge while both array pointers in range
    while (pa < ea && pb < eb) {
        *po++ = (*pa <= *pb) ? *pa++ : *pb++;
    }

    // Copy remaining elements from first array (if any)
    if (pa < ea) {
        size_t rem = (size_t)(ea - pa);
        memcpy(po, pa, rem * sizeof(int));
        po += rem;
    }
    
    // Copy remaining elements from second array (if any)
    if (pb < eb) {
        size_t rem = (size_t)(eb - pb);
        memcpy(po, pb, rem * sizeof(int));
    }
}

/**
 * Find partition point in arr1 for parallel merge
 * Use binary search to find where to split arr1 such that:
 * - i elements from arr1 + j elements from arr2 = target_pos total elements
 * - All elements in the partition are <= elements after the partition
 * 
 * This enables load-balanced parallel merging by finding the correct
 * split points that maintain sorted order.
 * 
 * @param arr1 //First sorted array
 * @param n1 //Length of first array
 * @param arr2 //Second sorted array
 * @param n2 //Length of second array
 * @param target_pos //Target position in merged output
 * @return //Index in arr1 for the partition
 */
static int find_partition_safe(const int *arr1, int n1, const int *arr2, int n2, int target_pos) {
    // Binary search bounds: ensure valid partition considering both array sizes
    int low = (target_pos - n2 > 0) ? (target_pos - n2) : 0;
    int high = (target_pos < n1) ? target_pos : n1;

    while (low < high) {
        int i = low + (high - low) / 2;  // Partition point in arr1
        int j = target_pos - i;           // Corresponding partition in arr2

        // Check if partition is valid
        if (i < n1 && j > 0 && arr1[i] < arr2[j - 1]) {
            low = i + 1;  // Need more elements from arr1
        } else {
            high = i;
        }
    }
    return low;
}

int main(int argc, char *argv[]) {
    int rank, size;      // MPI rank and total number of processes
    long long n1, n2;    // Size of input arrays

    // Parse command line arguments
    if (argc < 2) {
        if (rank == 0)
            printf("Usage: mpirun -np <procs> ./Parallel_merge <num_elements_per_array>\n");
        MPI_Finalize();
        return 1;
    }

    n1 = atoll(argv[1]);
    n2 = n1;   // Both arrays have the same size

    // Arrays for root process
    int *arr1 = NULL, *arr2 = NULL, *result = NULL;

    // Scatter/Gather metadata arrays (only needed on root)
    int *s_counts1 = NULL, *displs1 = NULL;   // For arr1 distribution
    int *s_counts2 = NULL, *displs2 = NULL;   // For arr2 distribution
    int *recv_counts = NULL, *displs_res = NULL; // For result gathering

    // Local arrays for each process
    int local_n1 = 0, local_n2 = 0;
    int *l_arr1 = NULL, *l_arr2 = NULL, *l_res = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process initializes data and computes work distribution
    if (rank == 0) {
        // Allocate input arrays
        arr1 = (int *)aligned_malloc((size_t)n1 * sizeof(int));
        arr2 = (int *)aligned_malloc((size_t)n2 * sizeof(int));
        if (!arr1 || !arr2) {
            fprintf(stderr, "Root: allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Running with %lld elements per array (%lld total)\n", n1, n1 + n2);

#if DO_GATHER_RESULT
        // Allocate result array if we're gathering
        result = (int *)aligned_malloc((size_t)(n1 + n2) * sizeof(int));
        if (!result) {
            fprintf(stderr, "Root: result allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#endif

        // Both are sorted, which simulates a realistic merge scenario
        for (long long i = 0; i < n1; i++) arr1[i] = (int)(i * 2);
        for (long long i = 0; i < n2; i++) arr2[i] = (int)(i * 2 + 1);

        // Allocate partition metadata arrays
        s_counts1 = (int *)malloc((size_t)size * sizeof(int));
        displs1   = (int *)malloc((size_t)size * sizeof(int));
        s_counts2 = (int *)malloc((size_t)size * sizeof(int));
        displs2   = (int *)malloc((size_t)size * sizeof(int));
        recv_counts = (int *)malloc((size_t)size * sizeof(int));
        displs_res  = (int *)malloc((size_t)size * sizeof(int));

        if (!s_counts1 || !displs1 || !s_counts2 || !displs2 || !recv_counts || !displs_res) {
            fprintf(stderr, "Root: partition arrays allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        long long total_ll = n1 + n2;

        // Check for MPI count limitations (32-bit integers)
        if (total_ll / size > INT_MAX) {
            if (rank == 0)
                printf("Error: Each process would exceed MPI 32-bit count limit.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int total = (int) total_ll;

        // Compute balanced work distribution
        const int chunk = total / size;
        const int remainder   = total % size;

        // For each process, determine what portions of arr1 and arr2 to send
        for (int r = 0; r < size; r++) {
            // Calculate the range [s_idx, e_idx) in the merged output for this rank
            const int s_idx = r * chunk + (r < remainder ? r : remainder);
            const int e_idx = s_idx + chunk + (r < remainder ? 1 : 0);

            // Find partition points: how many elements from arr1 vs arr2
            // to reach positions s_idx and e_idx in the merged result
            const int p1_s = find_partition_safe(arr1, (int)n1, arr2, (int)n2, s_idx);
            const int p1_e = find_partition_safe(arr1, (int)n1, arr2, (int)n2, e_idx);

            const int p2_s = s_idx - p1_s;  // Corresponding position in arr2
            const int p2_e = e_idx - p1_e;

            // Store scatter/gather parameters for this rank
            s_counts1[r] = p1_e - p1_s;  // Number of elements from arr1
            displs1[r]   = p1_s;          // arr1 starting position

            s_counts2[r] = p2_e - p2_s;  // Number of elements from arr2
            displs2[r]   = p2_s;          // arr2 starting position

            recv_counts[r] = e_idx - s_idx;  // Total elements this rank will produce
            displs_res[r]  = s_idx;           // Position in final result
        }
    }

    // Distribute work: each process learns its local array sizes
    MPI_Scatter(s_counts1, 1, MPI_INT, &local_n1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(s_counts2, 1, MPI_INT, &local_n2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local working arrays
    l_arr1 = (int *)aligned_malloc((size_t)local_n1 * sizeof(int));
    l_arr2 = (int *)aligned_malloc((size_t)local_n2 * sizeof(int));
    l_res  = (int *)aligned_malloc((size_t)(local_n1 + local_n2) * sizeof(int));

    if ((!l_arr1 && local_n1 > 0) || (!l_arr2 && local_n2 > 0) || (!l_res && (local_n1 + local_n2) > 0)) {
        fprintf(stderr, "Rank %d: local allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Distribute portions of arr1 and arr2 to all processes
    MPI_Scatterv(arr1, s_counts1, displs1, MPI_INT,
                 l_arr1, local_n1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(arr2, s_counts2, displs2, MPI_INT,
                 l_arr2, local_n2, MPI_INT, 0, MPI_COMM_WORLD);

    // Synchronize before timing the merge operation
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Each process independently merges its local portions
    sequential_merge_ptr(l_arr1, local_n1, l_arr2, local_n2, l_res);

#if TIME_MERGE_ONLY
    // Time only the merge operation (not the gather)
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
#else
    // Time includes everything up to gather
    double t1 = MPI_Wtime();
#endif

    // Gather all local results back to root
    // Possibly expensive so make it optional
#if DO_GATHER_RESULT
    MPI_Gatherv(l_res, local_n1 + local_n2, MPI_INT,
                result, recv_counts, displs_res, MPI_INT,
                0, MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Root process reports timing and validates result
    if (rank == 0) {
        if (TIME_MERGE_ONLY) {
            printf("Parallel local-merge time (barrier-bounded): %f s\n", t1 - t0);
        } else {
            printf("Time after scatter (includes merge pre-gather): %f s\n", t1 - t0);
        }

#if DO_GATHER_RESULT
        printf("Total time incl gather (barrier-bounded): %f s\n", t2 - t0);
        printf("Processing %lld total elements...\n", n1 + n2);

        // Verify the result is sorted
        int sorted = 1;
        for (long long i = 1; i < (n1 + n2); i++) {
            if (result[i] < result[i - 1]) { sorted = 0; break; }
        }
        printf("Sorted: %s\n", sorted ? "YES" : "NO");
#else
        printf("Result gather skipped (DO_GATHER_RESULT=0).\n");
#endif
    }

    // Clean up local arrays
    aligned_free(l_arr1);
    aligned_free(l_arr2);
    aligned_free(l_res);

    // Clean up root arrays
    if (rank == 0) {
        aligned_free(arr1);
        aligned_free(arr2);
#if DO_GATHER_RESULT
        aligned_free(result);
#endif
        free(s_counts1); free(displs1);
        free(s_counts2); free(displs2);
        free(recv_counts); free(displs_res);
    }

    MPI_Finalize();
    return 0;
}
