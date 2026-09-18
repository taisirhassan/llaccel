# Attention head widths and memory limits

Attention and RoPE support head widths 16, 32, 64, 128 and 256.
Local Verilator lint and all testbenches passed, including the wider heads at
context 4096. The rebuilt simulator also matched the integer reference on all
44 compiled Llama/Qwen test runs. See the [test report](../results/verilator-local-2026-09-15/README.md).

## Implemented changes

- Attention D decode and registers are nine bits; assertions check the complete instruction operand rather than a truncated value.
- Query, key/value assembly and output queues hold up to 256-byte rows. Four-bit piece indices and 16-bit request tags preserve row and byte offsets.
- The datapath retains 64 arithmetic lanes. A wide K row is assembled and consumed over two/four channel groups; only the complete dot product is published to softmax. The score/probability tile remains 256 entries.
- Value rows retain their normalized probability while updating the corresponding 64-channel accumulator groups. Output requantization still uses eight lanes per cycle.
- A pending-row gate prevents a new wide K/V row from overwriting active arithmetic. Up to 32 piece requests within the row use the existing request-tag FIFO.
- SRAM Q reads, KV_WRITE source reads and output/DRAM writes split wide rows into 16-byte pieces. Bus widths remain 64 bytes.
- RoPE uses nine-bit D and iterates one/one/two/four/eight 16-lane half-groups for D16/32/64/128/256, with group byte offset `group*32`.
- Attention and RoPE tests include D128/256. The attention harness now allocates persistent K/V in separate, dynamically sized DRAM fixtures, checks whole-memory integrity and rejects fixture-size underflow. These cases passed.

Per-head Q/K RMSNorm and projection biases already lower to existing device instructions. The previously added compiler and functional simulator support remains the independent software reference for comparison.

## Performance limitation

The conservative wide-head DRAM path reads a 64-byte beat for each 16-byte piece. Thus each complete wide row requests four times its useful data bytes, and wide rows are serialized through assembly/arithmetic. This is an explicit bandwidth/throughput tradeoff, not a measured performance result. Coalescing those piece reads is a possible later optimization. Existing performance measurements describe the earlier narrow-head implementation.

## Capacity limits remain

The resident RoPE table costs `2 * context * head_dim` bytes. D128/context4096 uses the entire 1 MiB scratchpad before activations; D256/context4096 uses 2 MiB. The compiler continues to reject configurations that do not fit. Table streaming/tiling is not implemented by this head-width extension.

The 32-bit DRAM address space also bounds complete checkpoint images. Support for the dense computation family does not imply every model size or context fits. Dynamic RoPE, active sliding windows, MoE and multimodal variants remain outside the documented contract.

