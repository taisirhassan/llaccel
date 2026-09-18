// llaccel_luts_undef.sv — clears the include guard of the generated
// rtl/llaccel_luts.svh between two modules that both `include it directly
// (vec_engine.sv, attn_engine.sv). Tools that preprocess the whole filelist as
// one unit (Verilator) keep macros across files, so without this the second
// include would be skipped and its tables undefined. Tools that treat every file
// as its own compilation unit (slang, yosys-slang) are unaffected.
`undef LLACCEL_LUTS_SVH
