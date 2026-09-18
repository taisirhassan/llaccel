// perf_counters.sv — NUM_PERF x 64-bit counters. Each counter adds its 16-bit
// increment every cycle (pulses are 0/1, byte counters are the bytes moved that
// cycle). clear (the start pulse) zeroes everything.
module perf_counters
  import llaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        clear,
  input  logic [15:0] inc  [NUM_PERF],
  output logic [63:0] perf [NUM_PERF]
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int unsigned i = 0; i < NUM_PERF; i++) perf[i] <= '0;
    end else begin
      for (int unsigned i = 0; i < NUM_PERF; i++) begin
        if (clear) perf[i] <= '0;
        else       perf[i] <= perf[i] + {48'd0, inc[i]};
      end
    end
  end
endmodule
