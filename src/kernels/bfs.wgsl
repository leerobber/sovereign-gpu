// BFS frontier expansion — one dispatch per level.
// Buffers:
//   0: adjacency offsets  u32[]  (CSR row_ptr, len = n_nodes+1)
//   1: adjacency cols     u32[]  (CSR col_idx)
//   2: visited            u32[]  (0/1, atomic)
//   3: frontier_in        u32[]  (current level node IDs, len=frontier_len)
//   4: frontier_out       u32[]  (next level node IDs, written atomically)
//   5: frontier_out_len   u32[1] (atomic counter for frontier_out)

@group(0) @binding(0) var<storage, read>            row_ptr:          array<u32>;
@group(0) @binding(1) var<storage, read>            col_idx:          array<u32>;
@group(0) @binding(2) var<storage, read_write>      visited:          array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>            frontier_in:      array<u32>;
@group(0) @binding(4) var<storage, read_write>      frontier_out:     array<u32>;
@group(0) @binding(5) var<storage, read_write>      frontier_out_len: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    // frontier_in[0] holds the count of active nodes.
    let n_frontier = frontier_in[0];
    if tid >= n_frontier { return; }

    let node  = frontier_in[tid + 1u];   // +1: slot 0 is the count
    let start = row_ptr[node];
    let end   = row_ptr[node + 1u];

    for (var i = start; i < end; i++) {
        let nbr = col_idx[i];
        // Try to claim the neighbour (0 → 1).
        let prev = atomicCompareExchangeWeak(&visited[nbr], 0u, 1u);
        if prev.old_value == 0u {
            let slot = atomicAdd(&frontier_out_len[0], 1u);
            frontier_out[slot + 1u] = nbr;   // +1: slot 0 reserved for count
        }
    }
}
