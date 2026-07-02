// Bellman-Ford single-source shortest paths (unweighted = weight 1).
// One dispatch per relaxation pass; host checks `changed` to detect convergence.
// Buffers:
//   0: row_ptr  u32[]
//   1: col_idx  u32[]
//   2: dist     u32[]   (distance from source; source initialised to 0, others to u32::MAX)
//   3: changed  u32[1]  (atomic flag; set to 1 if any dist updated this pass)

const INF: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read>        row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read>        col_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write>  dist:    array<u32>;
@group(0) @binding(3) var<storage, read_write>  changed: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    let n    = arrayLength(&dist);
    if node >= n { return; }

    let d_node = dist[node];
    if d_node == INF { return; }

    let start = row_ptr[node];
    let end   = row_ptr[node + 1u];
    let new_d = d_node + 1u;

    for (var i = start; i < end; i++) {
        let nbr = col_idx[i];
        if dist[nbr] > new_d {
            dist[nbr] = new_d;
            atomicStore(&changed[0], 1u);
        }
    }
}
