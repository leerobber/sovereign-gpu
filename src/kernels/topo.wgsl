// Kahn's algorithm — parallel in-degree decrement.
// Each dispatch processes one frontier of zero-in-degree nodes.
// Buffers:
//   0: row_ptr       u32[]   (CSR — outgoing edges)
//   1: col_idx       u32[]
//   2: in_degree     u32[]   (mutable; decremented as nodes are removed)
//   3: frontier_in   u32[]   (current zero-in-degree nodes; [0] = count)
//   4: frontier_out  u32[]   (next zero-in-degree nodes; [0] = count, atomic)
//   5: topo_order    u32[]   (output; written in order of removal)
//   6: topo_pos      u32[1]  (atomic write index into topo_order)

@group(0) @binding(0) var<storage, read>            row_ptr:      array<u32>;
@group(0) @binding(1) var<storage, read>            col_idx:      array<u32>;
@group(0) @binding(2) var<storage, read_write>      in_degree:    array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>            frontier_in:  array<u32>;
@group(0) @binding(4) var<storage, read_write>      frontier_out: array<u32>;
@group(0) @binding(5) var<storage, read_write>      topo_order:   array<u32>;
@group(0) @binding(6) var<storage, read_write>      topo_pos:     array<atomic<u32>>;

// atomic<u32> at binding 4 slot 0 — cast by host after dispatch
@group(0) @binding(7) var<storage, read_write>      fo_len:       array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid       = gid.x;
    let n_front   = frontier_in[0];
    if tid >= n_front { return; }

    let node  = frontier_in[tid + 1u];
    let pos   = atomicAdd(&topo_pos[0], 1u);
    topo_order[pos] = node;

    let start = row_ptr[node];
    let end   = row_ptr[node + 1u];

    for (var i = start; i < end; i++) {
        let nbr     = col_idx[i];
        let new_deg = atomicSub(&in_degree[nbr], 1u) - 1u;
        if new_deg == 0u {
            let slot = atomicAdd(&fo_len[0], 1u);
            frontier_out[slot + 1u] = nbr;
        }
    }
}
