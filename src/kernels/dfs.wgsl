// DFS post-order finish-time assignment (parallel label propagation).
// Each thread propagates labels from nodes to neighbours until stable.
// Buffers:
//   0: row_ptr   u32[]
//   1: col_idx   u32[]
//   2: label     u32[]   (node → finish-order label, initialised to node id)
//   3: changed   u32[1]  (set to 1 if any label updated this pass)

@group(0) @binding(0) var<storage, read>        row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read>        col_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write>  label:   array<u32>;
@group(0) @binding(3) var<storage, read_write>  changed: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node  = gid.x;
    let n     = arrayLength(&label);
    if node >= n { return; }

    let my_label = label[node];
    let start    = row_ptr[node];
    let end      = row_ptr[node + 1u];

    for (var i = start; i < end; i++) {
        let nbr = col_idx[i];
        if label[nbr] > my_label {
            label[nbr] = my_label;
            atomicStore(&changed[0], 1u);
        }
    }
}
