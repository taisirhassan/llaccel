// llaccel_luts_pkg.sv — wraps the generated ROM tables (rtl/llaccel_luts.svh) in a
// package so every engine can reference llaccel_luts_pkg::SIGMOID_LUT etc.
// The include guard is undefined again afterwards so a module that prefers to
// `include the .svh directly still works in single-compilation-unit tools.
package llaccel_luts_pkg;
`include "llaccel_luts.svh"
`undef LLACCEL_LUTS_SVH
endpackage
