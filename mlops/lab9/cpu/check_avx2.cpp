#include <iostream>
#include <bitset>
#include <stdexcept>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Function to execute CPUID instruction
void cpuid(int output[4], int function_id, int subfunction_id = 0) {
    __asm__ __volatile__(
        "cpuid" :
        "=a"(output[0]), "=b"(output[1]), "=c"(output[2]), "=d"(output[3]) :
        "a"(function_id), "c"(subfunction_id)
    );
}

// Check specific bit in a register
bool check_bit(int register_value, int bit_position) {
    return (register_value & (1 << bit_position)) != 0;
}

// Main function to check AVX2 support using CPUID
bool checkAVX2Support() {
    int info[4];
    cpuid(info, 0);  // Get the maximum supported CPUID function
    int max_function_id = info[0];

    if (max_function_id >= 7) {
        cpuid(info, 7);  // Extended features
        return check_bit(info[1], 5);  // Check EBX[5] for AVX2 support
    }
    return false;
}

#ifdef __APPLE__
// macOS-specific function to check system feature support via sysctl
bool checkOSXFeature(const std::string& feature) {
    char value[256];
    size_t size = sizeof(value);
    if (sysctlbyname(feature.c_str(), &value, &size, NULL, 0) == 0) {
        return std::string(value).find("avx2") != std::string::npos;
    }
    return false;
}
#endif

// Fallback or secondary check methods
bool secondaryAVX2Check() {
#ifdef __APPLE__
    return checkOSXFeature("machdep.cpu.leaf7_features");
#elif defined(__linux__)
    // Potential Linux-specific checks can be placed here
#else
    // Potential Windows-specific checks can be placed here using registry or other methods
#endif
    return false;
}

int main() {
    try {
        if (checkAVX2Support()) {
            std::cout << "AVX2 is supported on this CPU." << std::endl;
        } else if (secondaryAVX2Check()) {
            std::cout << "AVX2 is supported on this CPU (detected via secondary check)." << std::endl;
        } else {
            std::cout << "AVX2 is not supported on this CPU." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to detect AVX2 support: " << e.what() << std::endl;
    }
    return 0;
}
