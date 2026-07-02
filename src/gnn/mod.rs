/// GPU GNN embedding engine — real GraphSAGE-Mean dispatch via WGSL.
///
/// Computes per-node embeddings for the full graph in one kernel launch:
///   for each node u: agg = mean(neighbour features) → out = ReLU(agg × W)
///
/// Inputs (CSR layout):
///   row_ptr     — length n_nodes + 1
///   col_idx     — length n_edges
///   node_feats  — n_nodes × feat_dim, row-major
///   weight      — feat_dim × embed_dim, row-major
///
/// Output: n_nodes × embed_dim, row-major, flattened

#[cfg(feature = "webgpu")]
use crate::backends::webgpu::GpuDevice;

const WGSL_GNN_EMBED: &str = include_str!("../kernels/gnn_embed.wgsl");

// Inner struct keeps wgpu types behind the feature gate so the public
// struct compiles even without --features webgpu.
#[cfg(feature = "webgpu")]
struct GnnInner {
    dev:      GpuDevice,
    pipeline: wgpu::ComputePipeline,
}

pub struct GpuGnnEmbed {
    #[cfg(feature = "webgpu")]
    inner:    Option<GnnInner>,
    pub feat_dim:  usize,
    pub embed_dim: usize,
}

impl GpuGnnEmbed {
    /// Initialise the GNN engine.
    ///
    /// Returns None if no GPU adapter is available.
    /// feat_dim and embed_dim must match the WGSL kernel constants (32 / 128).
    pub fn new(feat_dim: usize, embed_dim: usize) -> Option<Self> {
        #[cfg(feature = "webgpu")]
        {
            let dev      = GpuDevice::new()?;
            let pipeline = dev.compile(WGSL_GNN_EMBED, "main");
            return Some(Self {
                inner: Some(GnnInner { dev, pipeline }),
                feat_dim,
                embed_dim,
            });
        }
        #[cfg(not(feature = "webgpu"))]
        Some(Self { feat_dim, embed_dim })
    }

    /// Compute embeddings for all nodes in one GPU dispatch.
    ///
    /// Returns a flattened Vec<f32> of length n_nodes × embed_dim.
    /// Falls back to zero vectors when no GPU device is present.
    pub fn embed(
        &self,
        row_ptr:    &[u32],
        col_idx:    &[u32],
        node_feats: &[f32],
        weight:     &[f32],
    ) -> Vec<f32> {
        let n_nodes = row_ptr.len().saturating_sub(1);
        if n_nodes == 0 {
            return vec![];
        }

        #[cfg(feature = "webgpu")]
        if let Some(inner) = &self.inner {
            let dev = &inner.dev;

            let b_feats  = dev.upload_f32(node_feats, "node_feats");
            let b_rptr   = dev.upload_u32(row_ptr,    "row_ptr");
            let b_cidx   = dev.upload_u32(col_idx,    "col_idx");
            let b_weight = dev.upload_f32(weight,     "weight");
            let b_out    = dev.alloc_f32(n_nodes * self.embed_dim, "out_embed");

            let bg = dev.bind(
                &inner.pipeline,
                &[&b_feats, &b_rptr, &b_cidx, &b_weight, &b_out],
            );
            let n_groups = ((n_nodes as u32) + 63) / 64;
            dev.dispatch(&inner.pipeline, &bg, n_groups);

            return dev.readback_f32(&b_out, n_nodes * self.embed_dim);
        }

        vec![0.0f32; n_nodes * self.embed_dim]
    }

    pub fn embed_dim(&self) -> usize { self.embed_dim }
    pub fn feat_dim(&self)  -> usize { self.feat_dim }
}
