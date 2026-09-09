#include "HIPSmith/HIPOutputMgr.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "ArrayVariable.h"
#include "Function.h"
#include "HIPSmith/HIPOptions.h"
#include "OutputMgr.h"
#include "Type.h"
#include "VariableSelector.h"

namespace HIPSmith {

HIPOutputMgr::HIPOutputMgr() : out_(HIPOptions::output()) {}

void HIPOutputMgr::OutputHeader(int argc, char *argv[], unsigned long seed) {
  std::ostream &out = get_main_out();

  out << "// "
         "------------------------------------------------------------------\n"
      << "// Standard Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include <cstdint>\n"
      << "#include <cstddef>\n"
      << "#include <climits>\n"
      << "#include <cstring>\n"
      << "#include <iostream>\n"
      << "#include <cstdint>\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// HIP Runtime\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include <hip/hip_runtime.h>\n\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// Project Headers\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#include \"HIPSmith.h\"\n"
      << "#include \"safe_math_macros.h\"\n\n"
      << "#define uint8_t unsigned char\n"
      << "#define int8_t char\n"

      << "// "
         "------------------------------------------------------------------\n"
      << "// Macros & Metadata\n"
      << "// "
         "------------------------------------------------------------------\n"
      << "#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, "
         "Y, Z)\n"
      << "// Seed: " << seed << "\n\n";
}

void HIPOutputMgr::Output() {
  // now will emit device coded functions
  HIPOptions::is_emitting_device_code(true);

  std::ostream &out = get_main_out();

  OutputStructUnionDeclarations(out);

  OutputHipGlobals();

  Globals *globals = Globals::GetGlobals();
  globals->OutputStructDefinition(out);
  globals->ModifyGlobalVariableReferences();
  globals->AddGlobalStructToAllFunctions();

  OutputForwardDeclarations(out);
  OutputFunctions(out);
  OutputEntryFunction(*globals);
}

std::ostream &HIPOutputMgr::get_main_out() { return out_; }

void HIPOutputMgr::OutputHipGlobals() {
  std::ostream &out = get_main_out();
  for (Variable *var : *VariableSelector::GetGlobalVariables()) {
    if (var->is_hip_builtin()) continue;

    // skip csmiths duplicate itemised arrays
    if (var->isArray) {
      ArrayVariable *av = dynamic_cast<ArrayVariable *>(var);
      if (av && !av->collective) continue;
    }

    // both var and device use the init line and then later copy semantics
    if (var->is_hip_const() || var->is_hip_device()) {
      // output the global defintion of the variable
      if (var->is_hip_const()) {
        out << "__constant__ ";
      } else {
        out << "__device__ ";
      }

      std::ostringstream oss_decl;
      var->OutputDecl(oss_decl);
      std::string decl_str = oss_decl.str();

      if (var->is_hip_const()) {
        size_t const_pos = decl_str.find("const ");
        if (const_pos != std::string::npos) {
          decl_str.erase(const_pos, 6); // Erase "const " (6 characters)
        }
      }

      out << decl_str << " = {};" << std::endl;

    } else if (var->is_hip_managed()) {
      // output global def of a managed variable
      out << "__managed__ ";
      var->OutputDecl(out);
      if (!(var->isArray &&
            static_cast<const ArrayVariable *>(var)->isVector)) {
        out << " = ";
      }

      // initialiser for the var
      if (var->isArray) {
        ArrayVariable *var_array = dynamic_cast<ArrayVariable *>(var);
        std::vector<std::string> init_strings;
        init_strings.push_back(var_array->init->to_string());
        for (const Expression *init : var_array->get_more_init_values()) {
          init_strings.push_back(init->to_string());
        }
        out << var_array->build_initializer_str(init_strings);
      } else {
        var->init->Output(out);
      }
      out << ";" << std::endl;
    }
  }
  out << std::endl;

  // we need to setup the HIP const special setup as cannot inline init them
  if (HIPOptions::hip_consts() || HIPOptions::hip_device()) {
    out << "void setup_hip_globals() {" << std::endl;

    for (Variable *var : *VariableSelector::GetGlobalVariables()) {
      // only do this delayed init for HIP const and HIP device memory
      if (!(var->is_hip_const() || var->is_hip_device())) continue;

      // do not use duplicated itemized arrays
      if (var->isArray) {
        ArrayVariable *av = dynamic_cast<ArrayVariable *>(var);
        if (av && !av->collective) continue;
      }

      output_tab(out, 2);

      // output a host version of the varaible that will then be copied to
      // device
      std::ostringstream oss;
      var->OutputDecl(oss);
      std::string decl_str = oss.str();

      // swap the actual name to a special host version
      size_t pos = decl_str.find(var->name);
      if (pos != std::string::npos) {
        decl_str.replace(pos, var->name.length(), "host_" + var->name);
      }
      out << decl_str << " = ";

      // initialiser for the var
      if (var->isArray) {
        ArrayVariable *var_array = dynamic_cast<ArrayVariable *>(var);
        std::vector<std::string> init_strings;
        init_strings.push_back(var_array->init->to_string());
        for (const Expression *init : var_array->get_more_init_values()) {
          init_strings.push_back(init->to_string());
        }
        out << var_array->build_initializer_str(init_strings);
      } else {
        var->init->Output(out);
      }
      out << ";" << std::endl;

      // copy from host to device
      output_tab(out, 2);
      out << "HIP_CHECK(hipMemcpyToSymbol(" << var->name << ", &host_"
          << var->name << ", sizeof(host_" << var->name << ")));" << std::endl;
    }
    out << "}" << std::endl << std::endl;
  }
}

void HIPOutputMgr::OutputEntryFunction(Globals &globals) {
  std::ostream &out = get_main_out();

  // ---------------------------------------------------------
  // Generated Kernel
  // ---------------------------------------------------------
  out << "// "
         "------------------------------------------------------------------"
      << std::endl;
  out << "// Kernel" << std::endl;
  out << "// "
         "------------------------------------------------------------------"
      << std::endl;
  out << "__global__ void hipsmith_kernel(uint64_t *results) {" << std::endl;

  output_tab(out, 1);
  out << "// Init globals" << std::endl;
  globals.OutputStructInit(out);
  out << std::endl;

  output_tab(out, 1);
  out << "// Execute core logic" << std::endl;
  output_tab(out, 1);
  out << "func_1(";
  globals.GetGlobalStructVar().Output(out);
  out << ");" << std::endl;
  out << std::endl;

  output_tab(out, 1);
  out << "// CRC Context" << std::endl;
  output_tab(out, 1);
  out << "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
  output_tab(out, 1);
  out << "int print_hash_value = 0;" << std::endl;

  output_tab(out, 1);
  out << "// Hash variables" << std::endl;
  HashGlobalVariables(out);
  out << std::endl;

  output_tab(out, 1);
  out << "// Store result" << std::endl;
  output_tab(out, 1);
  out << "int tid = threadIdx.x + blockIdx.x * blockDim.x;" << std::endl;
  output_tab(out, 1);
  out << "results[tid] = (crc64_context ^ 0xFFFFFFFFFFFFFFFFUL);" << std::endl;
  out << "}" << std::endl << std::endl;

  // ---------------------------------------------------------
  // Generated Main in its own file
  // ---------------------------------------------------------
  std::ofstream driver_out("HIP-driver.cpp");
  driver_out
      << "// "
         "------------------------------------------------------------------"
         "\n"
      << "// Host Main\n"
      << "// "
         "------------------------------------------------------------------"
         "\n"
      << "#include <hip/hip_runtime.h>\n"
      << "#include <iostream>\n"
      << "#include <cstdint>\n"
      << "#include <cstdio>\n"
      << "#include <vector>\n"
      << "#include \"HIPSmith.h\"\n"
      << "extern __global__ void hipsmith_kernel(uint64_t *results);\n";

  if (HIPSmith::HIPOptions::hip_consts() ||
      HIPSmith::HIPOptions::hip_device()) {
    driver_out << "extern void setup_hip_globals();\n";
  }

  driver_out << "\nint main(int argc, const char* argv[]) {\n";

  if (HIPSmith::HIPOptions::hip_managed() || HIPOptions::hip_device() ||
      HIPOptions::hip_builtins() || HIPOptions::hip_sync() ||
      HIPOptions::hip_warp() || HIPOptions::hip_warp_match() ||
      HIPOptions::hip_warp_shuffle() || HIPOptions::hip_warp_reduce() ||
      HIPOptions::hip_atomic() || HIPOptions::hip_argc_threads()) {
    output_tab(driver_out, 1);
    driver_out << "// argc should be 1 so block size=1 enforced due to using "
                  "HIP's managed memory OR HIP device memory OR hip builtins "
                  "like threadID or thread sync primitives or warp level stuff or atomics"
               << std::endl;
    output_tab(driver_out, 1);
    driver_out << "const unsigned int num_threads = argc;" << std::endl;
  } else {
    output_tab(driver_out, 1);
    driver_out << "const unsigned int num_threads = 4;" << std::endl;
  }

  // HIP shared memory requires block size is 1 or we will get data races
  // across the block
  if (HIPSmith::HIPOptions::hip_shared() ||
      HIPSmith::HIPOptions::hip_managed() || HIPOptions::hip_device() ||
      HIPOptions::hip_builtins() || HIPOptions::hip_sync() ||
      HIPOptions::hip_warp() || HIPOptions::hip_warp_match() ||
      HIPOptions::hip_warp_shuffle() || HIPOptions::hip_warp_reduce() ||
      HIPOptions::hip_atomic() || HIPOptions::hip_argc_threads()) {
    output_tab(driver_out, 1);
    driver_out << "// argc should be 1 so block size=1 enforced due to using "
                  "HIP's shared memory, device memory or managed memory OR hip "
                  "builtins like threadID or thread sync primitives or warp "
                  "level stuff"
               << std::endl;
    output_tab(driver_out, 1);
    driver_out << "const unsigned int block_size = argc;" << std::endl;
  } else {
    output_tab(driver_out, 1);
    driver_out << "const unsigned int block_size = 4;" << std::endl;
  }

  output_tab(driver_out, 1);
  driver_out << "// Host Alloc" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "std::vector<uint64_t> h_results(num_threads);" << std::endl;
  output_tab(driver_out, 1);
  driver_out
      << "const size_t results_bytes = sizeof(uint64_t) * h_results.size();"
      << std::endl;
  driver_out << std::endl;

  if (HIPSmith::HIPOptions::hip_consts() || HIPOptions::hip_device()) {
    output_tab(driver_out, 1);
    driver_out << "setup_hip_globals();" << std::endl;
    driver_out << std::endl;
  }

  output_tab(driver_out, 1);
  driver_out << "// Device Alloc" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "uint64_t *d_results;" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "HIP_CHECK(hipMalloc((void**)&d_results, results_bytes));"
             << std::endl;
  output_tab(driver_out, 1);
  driver_out << "HIP_CHECK(hipMemset(d_results, 0, results_bytes));"
             << std::endl;
  driver_out << std::endl;

  output_tab(driver_out, 1);
  driver_out << "// Dimensions" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "const dim3 block_dim(block_size);" << std::endl;
  output_tab(driver_out, 1);
  driver_out
      << "const dim3 grid_dim((num_threads + block_size - 1) / block_size);"
      << std::endl;
  driver_out << std::endl;

  output_tab(driver_out, 1);
  driver_out << "// Launch" << std::endl;
  output_tab(driver_out, 1);
  driver_out
      << "hipLaunchKernelGGL(hipsmith_kernel, grid_dim, block_dim, 0, 0, "
         "d_results);"
      << std::endl;
  output_tab(driver_out, 1);
  driver_out << "HIP_CHECK(hipGetLastError());" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "HIP_CHECK(hipDeviceSynchronize());" << std::endl;
  driver_out << std::endl;

  output_tab(driver_out, 1);
  driver_out << "// Copy Back" << std::endl;
  output_tab(driver_out, 1);
  driver_out
      << "HIP_CHECK(hipMemcpy(h_results.data(), d_results, results_bytes, "
         "hipMemcpyDeviceToHost));"
      << std::endl;
  driver_out << std::endl;

  output_tab(driver_out, 1);
  driver_out << "// Free" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "HIP_CHECK(hipFree(d_results));" << std::endl;
  driver_out << std::endl;

  output_tab(driver_out, 1);
  driver_out << "// Output" << std::endl;
  output_tab(driver_out, 1);
  driver_out << "for (size_t i = 0; i < h_results.size(); ++i) {" << std::endl;
  output_tab(driver_out, 2);
  driver_out << "printf(\"Thread %zu CRC: %lu\\n\", i, h_results[i]);"
             << std::endl;
  output_tab(driver_out, 1);
  driver_out << "}" << std::endl;

  output_tab(driver_out, 1);
  driver_out << "return 0;" << std::endl;
  driver_out << "}" << std::endl;
}
}  // namespace HIPSmith