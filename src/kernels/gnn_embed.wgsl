// GraphSAGE-style node embedding kernel.
// For each node: aggregate (mean) neighbour features → linear projection → ReLU.
//
// Buffers:
//   0: node_feats   f32[]   (n_nodes × feat_dim, row-major input features)
//   1: row_ptr      u32[]   (CSR adjacency row pointers)
//   2: col_idx      u32[]   (CSR adjacency column indices)
//   3: weight       f32[]   (feat_dim × embed_dim projection matrix)
//   4: out_embed    f32[]   (n_nodes × embed_dim, output embeddings)
//
// TODO: wire buffer layout and activate this kernel.

// Placeholder — real shader will use @group/@binding and actual dimensions.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // node = gid.x
    // 1. Aggregate: mean of neighbour features + self
    // 2. Project: out = W * agg
    // 3. Activate: out = max(0, out)
    // (not yet implemented)
    let _ = gid;
}
