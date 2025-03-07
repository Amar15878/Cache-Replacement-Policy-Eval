// Author: Amar Chouhan and Mohamed Habib Loukil

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <queue>
#include <limits>
#include <cstdint>
#include <complex>
#include <cmath>

using namespace std;

// Constants for cache simulation
const uint64_t CACHE_LINE_SIZE = 64; // Size of a cache line in bytes

// Helper function to map addresses to cache lines
inline uint64_t getCacheLine(uint64_t address) {
    return address / CACHE_LINE_SIZE;
}

// Base class for Replacement Policies
class ReplacementPolicy {
public:
    virtual bool access(uint64_t address) = 0; // Return true if hit, false if miss
    virtual void reset() = 0;
    virtual ~ReplacementPolicy() {}
};

// LRU Replacement Policy
class LRUCache : public ReplacementPolicy {
public:
    LRUCache(size_t capacity) : capacity(capacity) {}

    bool access(uint64_t address) override {
        uint64_t cacheLine = getCacheLine(address);
        bool hit = (addressMap.find(cacheLine) != addressMap.end());
        if (hit) {
            usageOrder.erase(addressMap[cacheLine]);
        }
        else if (usageOrder.size() >= capacity) {
            uint64_t lru = usageOrder.back();
            usageOrder.pop_back();
            addressMap.erase(lru);
        }
        usageOrder.push_front(cacheLine);
        addressMap[cacheLine] = usageOrder.begin();
        return hit;
    }

    void reset() override {
        usageOrder.clear();
        addressMap.clear();
    }

private:
    size_t capacity;
    list<uint64_t> usageOrder;
    unordered_map<uint64_t, list<uint64_t>::iterator> addressMap;
};

// LFU Replacement Policy
class LFUCache : public ReplacementPolicy {
public:
    LFUCache(size_t capacity) : capacity(capacity) {}

    bool access(uint64_t address) override {
        uint64_t cacheLine = getCacheLine(address);
        frequencyMap[cacheLine]++;
        bool hit = (cacheItems.find(cacheLine) != cacheItems.end());
        if (!hit) {
            if (cacheItems.size() >= capacity) {
                evict();
            }
            cacheItems.insert(cacheLine);
        }
        return hit;
    }

    void reset() override {
        cacheItems.clear();
        frequencyMap.clear();
    }

private:
    size_t capacity;
    unordered_set<uint64_t> cacheItems;
    unordered_map<uint64_t, int> frequencyMap;

    void evict() {
        uint64_t lfu = 0;
        int minFreq = INT_MAX;
        for (uint64_t cacheLine : cacheItems) {
            if (frequencyMap[cacheLine] < minFreq) {
                minFreq = frequencyMap[cacheLine];
                lfu = cacheLine;
            }
        }
        cacheItems.erase(lfu);
        // Do not erase frequencyMap[lfu]; keep frequencies cumulative
    }
};

// MRU Replacement Policy
class MRUCache : public ReplacementPolicy {
public:
    MRUCache(size_t capacity) : capacity(capacity) {}

    bool access(uint64_t address) override {
        uint64_t cacheLine = getCacheLine(address);
        bool hit = (cacheItems.find(cacheLine) != cacheItems.end());
        if (hit) {
            usageStack.erase(addressMap[cacheLine]);
        }
        else if (usageStack.size() >= capacity) {
            uint64_t mru = usageStack.front();
            usageStack.pop_front();
            cacheItems.erase(mru);
            addressMap.erase(mru);
        }
        usageStack.push_front(cacheLine);
        addressMap[cacheLine] = usageStack.begin();
        cacheItems.insert(cacheLine);
        return hit;
    }

    void reset() override {
        usageStack.clear();
        cacheItems.clear();
        addressMap.clear();
    }

private:
    size_t capacity;
    list<uint64_t> usageStack;
    unordered_set<uint64_t> cacheItems;
    unordered_map<uint64_t, list<uint64_t>::iterator> addressMap;
};

// Mockingjay Cache Replacement Policy with Optimized Eviction
class MockingjayCache : public ReplacementPolicy {
public:
    MockingjayCache(size_t capacity, double alpha)
        : capacity(capacity), alpha(alpha), max_frequency(0), max_recency(0), currentRecency(0) {}

    bool access(uint64_t address) override {
        uint64_t cacheLine = getCacheLine(address);

        // Update frequency
        frequencyTable[cacheLine]++;
        max_frequency = std::max(max_frequency, frequencyTable[cacheLine]);

        // Update recency
        recencyTable[cacheLine] = currentRecency++;
        max_recency = std::max(max_recency, recencyTable[cacheLine]);

        // Check if the block is already in the cache
        bool hit = (cacheEntries.find(cacheLine) != cacheEntries.end());
        if (hit) {
            // Update the min-heap score
            double score = calculateScore(frequencyTable[cacheLine], recencyTable[cacheLine]);
            minHeap.push({ score, cacheLine });
        }
        else {
            // Evict if the cache is full
            if (cacheEntries.size() >= capacity) {
                evict();
            }
            // Add the new block to the cache
            cacheEntries.insert(cacheLine);
            double score = calculateScore(frequencyTable[cacheLine], recencyTable[cacheLine]);
            minHeap.push({ score, cacheLine });
        }
        return hit;
    }

    void reset() override {
        cacheEntries.clear();
        frequencyTable.clear();
        recencyTable.clear();
        while (!minHeap.empty()) {
            minHeap.pop();
        }
        max_frequency = 0;
        max_recency = 0;
        currentRecency = 0;
    }

private:
    size_t capacity;                                  // Cache capacity
    double alpha;                                     // Adaptation parameter (0 = pure LRU, 1 = pure LFU)
    unordered_set<uint64_t> cacheEntries;            // Current cache entries
    unordered_map<uint64_t, int> frequencyTable;     // Frequency table
    unordered_map<uint64_t, uint64_t> recencyTable;  // Recency table
    uint64_t currentRecency;                         // Global recency counter
    int max_frequency;                               // Maximum frequency in the cache
    uint64_t max_recency;                            // Maximum recency value in the cache

    // Min-heap to track entries with minimum score
    typedef pair<double, uint64_t> HeapEntry;        // {score, cacheLine}
    struct CompareHeapEntry {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const {
            return a.first > b.first; // Min-heap
        }
    };
    priority_queue<HeapEntry, vector<HeapEntry>, CompareHeapEntry> minHeap;

    // Calculate the score for eviction (normalized frequency and recency)
    double calculateScore(int frequency, uint64_t recency) {
        double frequency_normalized = (max_frequency > 0) ? static_cast<double>(frequency) / max_frequency : 0;
        double recency_normalized = (max_recency > 0) ? static_cast<double>(recency) / max_recency : 0;
        return alpha * frequency_normalized + (1.0 - alpha) * recency_normalized;
    }

    void evict() {
        while (!minHeap.empty()) {
            uint64_t cacheLine = minHeap.top().second;
            minHeap.pop();
            // Verify that the cache line is still valid
            if (cacheEntries.find(cacheLine) != cacheEntries.end()) {
                cacheEntries.erase(cacheLine);
                // Do not erase frequencyTable[cacheLine]; keep frequencies cumulative
                recencyTable.erase(cacheLine);
                break;
            }
        }
    }
};

// Cache Simulator
class Cache {
public:
    Cache(size_t cacheSize, ReplacementPolicy* policy)
        : replacementPolicy(policy), missCount(0), accessCount(0) {}

    void accessMemory(uint64_t address) {
        accessCount++;
        if (!replacementPolicy->access(address)) {
            missCount++;
        }
    }

    double getMissRate() const {
        return (double)missCount / accessCount * 100.0;
    }

    void resetStats() {
        missCount = 0;
        accessCount = 0;
        replacementPolicy->reset();
    }

    ~Cache() {
        delete replacementPolicy;
    }

    // Add public getter methods for missCount and accessCount
    size_t getMissCount() const {
        return missCount;
    }

    size_t getAccessCount() const {
        return accessCount;
    }

private:
    ReplacementPolicy* replacementPolicy;
    size_t missCount;
    size_t accessCount;
};

// Matrix Multiplication
void matrixMultiply(vector<vector<int>>& A, vector<vector<int>>& B, vector<vector<int>>& C, int N, Cache& cache) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            cache.accessMemory(reinterpret_cast<uint64_t>(&C[i][j]));
            for (int k = 0; k < N; ++k) {
                cache.accessMemory(reinterpret_cast<uint64_t>(&A[i][k]));
                cache.accessMemory(reinterpret_cast<uint64_t>(&B[k][j]));
                C[i][j] += A[i][k] * B[k][j];
                cache.accessMemory(reinterpret_cast<uint64_t>(&C[i][j]));
            }
        }
    }
}

// Insertion Sort
void insertionSort(vector<int>& arr, Cache& cache) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        cache.accessMemory(reinterpret_cast<uint64_t>(&arr[i]));
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            cache.accessMemory(reinterpret_cast<uint64_t>(&arr[j]));
            arr[j + 1] = arr[j];
            cache.accessMemory(reinterpret_cast<uint64_t>(&arr[j + 1]));
            --j;
        }
        arr[j + 1] = key;
        cache.accessMemory(reinterpret_cast<uint64_t>(&arr[j + 1]));
    }
}

// Discrete Fourier Transform (DFT)
void discreteFourierTransform(vector<complex<double>>& input, vector<complex<double>>& output, int N, Cache& cache) {
    const double PI = 3.14159265358979323846;
    for (int k = 0; k < N; ++k) {
        output[k] = complex<double>(0, 0);
        cache.accessMemory(reinterpret_cast<uint64_t>(&output[k])); // Access output[k] (write)
        for (int n = 0; n < N; ++n) {
            cache.accessMemory(reinterpret_cast<uint64_t>(&input[n])); // Access input[n] (read)
            double angle = -2 * PI * k * n / N;
            output[k] += input[n] * complex<double>(cos(angle), sin(angle));
            cache.accessMemory(reinterpret_cast<uint64_t>(&output[k])); // Access output[k] (write)
        }
    }
}

// Fast Fourier Transform (FFT)
void fastFourierTransform(vector<complex<double>>& input, vector<complex<double>>& output, int N, Cache& cache) {
    const double PI = 3.14159265358979323846;
    if (N == 1) {
        output[0] = input[0];
        cache.accessMemory(reinterpret_cast<uint64_t>(&input[0]));
        cache.accessMemory(reinterpret_cast<uint64_t>(&output[0]));
        return;
    }
    vector<complex<double>> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
        cache.accessMemory(reinterpret_cast<uint64_t>(&input[2 * i]));
        cache.accessMemory(reinterpret_cast<uint64_t>(&input[2 * i + 1]));
    }
    vector<complex<double>> evenOutput(N / 2), oddOutput(N / 2);
    fastFourierTransform(even, evenOutput, N / 2, cache);
    fastFourierTransform(odd, oddOutput, N / 2, cache);
    for (int k = 0; k < N / 2; ++k) {
        cache.accessMemory(reinterpret_cast<uint64_t>(&evenOutput[k]));
        cache.accessMemory(reinterpret_cast<uint64_t>(&oddOutput[k]));
        complex<double> t = polar(1.0, -2 * PI * k / N) * oddOutput[k];
        output[k] = evenOutput[k] + t;
        output[k + N / 2] = evenOutput[k] - t;
        cache.accessMemory(reinterpret_cast<uint64_t>(&output[k]));
        cache.accessMemory(reinterpret_cast<uint64_t>(&output[k + N / 2]));
    }
}

// Main Function
int main() {
    const size_t CACHE_SIZE = 256;  // Cache size (number of cache lines)
    const int MATRIX_SIZE = 64;      // Matrix size
    const int ARRAY_SIZE = 2048;      // Array size
    const int FFT_SIZE = 1024;         // Size for DFT/FFT
    const string TRACE_FILE = "gcc.txt"; // Trace file name

    // Initialize matrices and arrays with deterministic data
    vector<vector<int>> A(MATRIX_SIZE, vector<int>(MATRIX_SIZE));
    vector<vector<int>> B(MATRIX_SIZE, vector<int>(MATRIX_SIZE));
    vector<vector<int>> C(MATRIX_SIZE, vector<int>(MATRIX_SIZE, 0));

    // Fill matrices A and B with deterministic values
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            A[i][j] = (i + j) % 10; // Deterministic value
            B[i][j] = (i * j) % 10; // Deterministic value
        }
    }

    // Initialize array for insertion sort
    vector<int> arr(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = ARRAY_SIZE - i; // Descending order
    }

    // Initialize input for DFT and FFT
    vector<complex<double>> input(FFT_SIZE), output(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; ++i) {
        input[i] = complex<double>(i % 10, (FFT_SIZE - i) % 10); // Deterministic values
    }

    // List of cache replacement policies to test
    vector<string> policyNames = { "LRU", "LFU", "MRU", "Mockingjay" };

    // Loop over each policy
    for (const auto& policyName : policyNames) {
        ReplacementPolicy* policy = nullptr;
        if (policyName == "LRU") {
            policy = new LRUCache(CACHE_SIZE);
        }
        else if (policyName == "LFU") {
            policy = new LFUCache(CACHE_SIZE);
        }
        else if (policyName == "MRU") {
            policy = new MRUCache(CACHE_SIZE);
        }
        else if (policyName == "Mockingjay") {
            policy = new MockingjayCache(CACHE_SIZE, 0.1);
        }

        if (policy != nullptr) {
            Cache cache(CACHE_SIZE, policy);

            // Process the gcc.txt trace file
            cache.resetStats();
            ifstream traceFile(TRACE_FILE);
            if (!traceFile.is_open()) {
                cerr << "Error: Could not open trace file " << TRACE_FILE << endl;
                return 1;
            }

            string line;
            while (getline(traceFile, line)) {
                // Remove any leading/trailing whitespace
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);

                if (line.empty()) continue; // Skip empty lines

                // Parse the operation and address
                stringstream ss(line);
                char operation;
                string addressStr;
                ss >> operation >> addressStr;

                // Convert address from hex string to integer
                uint64_t address;
                try {
                    address = stoull(addressStr, nullptr, 16);
                }
                catch (const exception& e) {
                    cerr << "Error parsing address: " << addressStr << endl;
                    continue;
                }

                // Process the memory access
                cache.accessMemory(address);
            }

            traceFile.close();

            double traceMissRate = cache.getMissRate();

            // Now test the algorithms
            // Matrix Multiplication
            cache.resetStats();
            matrixMultiply(A, B, C, MATRIX_SIZE, cache);
            double matrixMissRate = cache.getMissRate();

            // Insertion Sort
            cache.resetStats();
            insertionSort(arr, cache);
            double sortMissRate = cache.getMissRate();

            // DFT
            cache.resetStats();
            discreteFourierTransform(input, output, FFT_SIZE, cache);
            double dftMissRate = cache.getMissRate();

            // FFT
            cache.resetStats();
            fastFourierTransform(input, output, FFT_SIZE, cache);
            double fftMissRate = cache.getMissRate();

            // Display Results
            cout << "Policy: " << policyName << endl;
            cout << "Trace Miss Rate: " << traceMissRate << "%" << endl;
            cout << "Matrix Multiplication Miss Rate: " << matrixMissRate << "%" << endl;
            cout << "Insertion Sort Miss Rate: " << sortMissRate << "%" << endl;
            cout << "DFT Miss Rate: " << dftMissRate << "%" << endl;
            cout << "FFT Miss Rate: " << fftMissRate << "%" << endl;
            cout << "--------------------------------------------" << endl;
        }
    }

    return 0;
}
