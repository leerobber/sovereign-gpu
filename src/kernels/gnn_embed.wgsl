// GraphSAGE-Mean node embedding kernel.
// For each node u: aggregate mean of neighbour features, project via W, activate (ReLU).
//
// Buffers:
//   0: node_feats  f32[n_nodes * feat_dim]   — row-major input features
//   1: row_ptr     u32[n_nodes + 1]           — CSR row pointers
//   2: col_idx     u32[n_edges]               — CSR column indices
//   3: weight      f32[feat_dim * embed_dim]  — projection matrix W (feat_dim rows)
//   4: out_embed   f32[n_nodes * embed_dim]   — output embeddings (read_write)

struct NodeFeats { data: array<f32> };
struct RowPtr    { data: array<u32> };
struct ColIdx    { data: array<u32> };
struct Weight    { data: array<f32> };
struct OutEmbed  { data: array<f32> };

@group(0) @binding(0) var<storage, read>       node_feats : NodeFeats;
@group(0) @binding(1) var<storage, read>       row_ptr    : RowPtr;
@group(0) @binding(2) var<storage, read>       col_idx    : ColIdx;
@group(0) @binding(3) var<storage, read>       weight     : Weight;
@group(0) @binding(4) var<storage, read_write> out_embed  : OutEmbed;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let u        = gid.x;
    let feat_dim  = 32u;
    let embed_dim = 128u;

    // 1. Mean-aggregate neighbour features into agg[feat_dim]
    var agg: array<f32, 32>;
    let start = row_ptr.data[u];
    let end   = row_ptr.data[u + 1u];
    let deg   = end - start;

    if deg > 0u {
        for (var e = start; e < end; e++) {
            let v    = col_idx.data[e];
            let base = v * feat_dim;
            for (var i = 0u; i < feat_dim; i++) {
                agg[i] += node_feats.data[base + i];
            }
        }
        for (var i = 0u; i < feat_dim; i++) {
            agg[i] = agg[i] / f32(deg);
        }
    } else {
        // isolated node: use own features as aggregate
        let base = u * feat_dim;
        for (var i = 0u; i < feat_dim; i++) {
            agg[i] = node_feats.data[base + i];
        }
    }

    // 2. Linear projection: out[j] = ReLU(sum_i agg[i] * W[i, j])
    let out_base = u * embed_dim;
    for (var j = 0u; j < embed_dim; j++) {
        var sum = 0.0f;
        for (var i = 0u; i < feat_dim; i++) {
            sum += agg[i] * weight.data[i * embed_dim + j];
        }
        out_embed.data[out_base + j] = max(0.0f, sum);
    }
}
