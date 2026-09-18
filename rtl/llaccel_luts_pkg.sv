// llaccel_luts_pkg.sv — wraps the generated ROM tables (rtl/llaccel_luts.svh) in a
// package. The engines reference llaccel_luts_pkg::SIGMOID_LUT / EXP_INT_LUT /
// EXP_FRAC_LUT (gemm_engine by qualified name, vec_engine / attn_engine by
// importing the package). Nothing else includes the .svh, so the include guard
// stays intact in single-compilation-unit tools (Verilator, slang, yosys-slang).
package llaccel_luts_pkg;
`include "llaccel_luts.svh"
endpackage
