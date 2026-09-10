// sync_fifo.sv — generic synchronous FIFO (registered storage, combinational head).
// push is ignored when full, pop is ignored when empty; callers gate on full/empty.
// count is the number of valid entries; clr flushes synchronously (used by start).
module sync_fifo #(
  parameter int unsigned WIDTH = 8,
  parameter int unsigned DEPTH = 4
) (
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       clr,
  input  logic                       push,
  input  logic [WIDTH-1:0]           wdata,
  input  logic                       pop,
  output logic [WIDTH-1:0]           rdata,
  output logic                       full,
  output logic                       empty,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  localparam int unsigned AW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
  localparam int unsigned CW = $clog2(DEPTH + 1);

  logic [WIDTH-1:0] mem [DEPTH];
  logic [AW-1:0]    wp, rp;
  logic             do_push, do_pop;

  assign full    = (count == CW'(DEPTH));
  assign empty   = (count == '0);
  assign do_push = push && !full;
  assign do_pop  = pop && !empty;
  assign rdata   = mem[rp];

  always_ff @(posedge clk) begin
    if (do_push) mem[wp] <= wdata;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || clr) begin
      wp    <= '0;
      rp    <= '0;
      count <= '0;
    end else begin
      if (do_push) wp <= (wp == AW'(DEPTH - 1)) ? '0 : wp + 1'b1;
      if (do_pop)  rp <= (rp == AW'(DEPTH - 1)) ? '0 : rp + 1'b1;
      case ({do_push, do_pop})
        2'b10:   count <= count + 1'b1;
        2'b01:   count <= count - 1'b1;
        default: count <= count;
      endcase
    end
  end
endmodule
