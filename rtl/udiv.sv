// udiv.sv — restoring unsigned integer divider, floor(a / b).
//
// Parameterized widths: AW-bit dividend, BW-bit divisor, AW-bit quotient.
// One quotient bit per cycle, MSB first (AW iteration cycles). A zero divisor
// is treated as 1 (quotient = a), matching numerics.h::udiv.
//
// Timing: `start` (with `a`, `b`) is sampled on a clock edge E0; the AW
// iterations happen on edges E1..E_AW and `done` / `q` are registered at E_AW,
// so `done` is high for the one cycle that follows E_AW (AW cycles after E0;
// `busy` is high for those AW cycles). `q` holds until the next `start`.
// `start` is ignored while `busy`.
module udiv #(
  parameter int AW = 32,
  parameter int BW = 24
) (
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,
  input  logic [AW-1:0] a,
  input  logic [BW-1:0] b,
  output logic          busy,
  output logic          done,
  output logic [AW-1:0] q
);
  localparam int CW = (AW > 1) ? $clog2(AW + 1) : 1;

  logic [BW-1:0] rem;      // partial remainder (< divisor after each step)
  logic [BW-1:0] dvs;      // divisor (0 replaced by 1)
  logic [AW-1:0] num;      // dividend, shifted left one bit per iteration
  logic [AW-1:0] quo;      // quotient bits so far (shift form supports AW = 1 too)
  logic [CW-1:0] cnt;      // iterations remaining

  logic [BW:0]   rem_sh;   // remainder with the next dividend bit appended
  logic          take;

  always_comb begin
    rem_sh = {rem, num[AW-1]};
    take   = (rem_sh >= {1'b0, dvs});
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0;
      done <= 1'b0;
      rem  <= '0;
      dvs  <= '0;
      num  <= '0;
      quo  <= '0;
      cnt  <= '0;
      q    <= '0;
    end else begin
      done <= 1'b0;
      if (!busy) begin
        if (start) begin
          busy <= 1'b1;
          rem  <= '0;
          dvs  <= (b == '0) ? {{(BW-1){1'b0}}, 1'b1} : b;
          num  <= a;
          quo  <= '0;
          cnt  <= CW'(AW);
        end
      end else begin
        rem <= take ? BW'(rem_sh - {1'b0, dvs}) : BW'(rem_sh);  // < dvs < 2^BW
        num <= num << 1;
        quo <= (quo << 1) | AW'(take);
        cnt <= cnt - CW'(1);
        if (cnt == CW'(1)) begin
          busy <= 1'b0;
          done <= 1'b1;
          q    <= (quo << 1) | AW'(take);
        end
      end
    end
  end
endmodule
