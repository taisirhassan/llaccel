// isqrt.sv — bit-serial floor(sqrt(a)) for a 48-bit unsigned operand.
//
// Exactly the algorithm of include/llaccel/numerics.h::isqrt48: 24 iterations,
// each consuming two operand bits (MSB first), doubling the partial root and
// conditionally subtracting the trial value (2*root + 1) from the remainder.
//
// Timing: `start` (with `a`) is sampled on a clock edge E0; the 24 iterations
// happen on edges E1..E24 and `done` / `q` are registered at E24, so `done` is
// high for the one cycle that follows E24 (24 cycles after E0; `busy` is high
// for those 24 cycles). `q` holds until the next `start`. `start` is ignored
// while `busy`.
module isqrt (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [47:0] a,
  output logic        busy,
  output logic        done,
  output logic [23:0] q
);
  // Invariant after each step: rem <= 2*root < 2^25 (25 bits). Before the
  // subtraction the shifted remainder (rem << 2 | 2 bits) is < 2^27 (27 bits).
  // The trial value is (2*root_new + 1) with root_new = 2*root, i.e. 4*root + 1
  // in terms of the partial root held in `root` (numerics.h doubles the root
  // before forming the trial).
  logic [24:0] rem;
  logic [23:0] root;
  logic [47:0] t;        // operand, shifted left by two each iteration
  logic [4:0]  cnt;      // iterations remaining

  logic [26:0] rem_sh;   // remainder with the next two operand bits appended
  logic [26:0] trial;    // (root << 2) | 1, zero-extended
  logic        take;

  always_comb begin
    rem_sh = {rem, t[47:46]};
    trial  = {1'b0, root, 2'b01};
    take   = (trial <= rem_sh);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0;
      done <= 1'b0;
      rem  <= '0;
      root <= '0;
      t    <= '0;
      cnt  <= '0;
      q    <= '0;
    end else begin
      done <= 1'b0;
      if (!busy) begin
        if (start) begin
          busy <= 1'b1;
          rem  <= '0;
          root <= '0;
          t    <= a;
          cnt  <= 5'd24;
        end
      end else begin
        rem  <= take ? 25'(rem_sh - trial) : 25'(rem_sh);   // result < 2^25 by the invariant
        root <= {root[22:0], take};
        t    <= {t[45:0], 2'b00};
        cnt  <= cnt - 5'd1;
        if (cnt == 5'd1) begin
          busy <= 1'b0;
          done <= 1'b1;
          q    <= {root[22:0], take};
        end
      end
    end
  end
endmodule
