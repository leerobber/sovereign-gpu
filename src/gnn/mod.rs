/// GPU GNN embedding engine.
///
/// Computes per-node embeddings via a GraphSAGE-style WGSL kernel:
/// for each node, aggregate neighbour features, project, activate.
/// Current implementation returns zero-vectors pending WGSL kernel authoring.

#[cfg(feature = "webgpu")]
use crate::backends::webgpu::GpuDevice;

const WGSL_GNN_EMBED: &str = include_str!("../kernels/gnn_embed.wgsl");
const EMBED_DIM: usize = 128;

pub struct GpuGnnEmbed {
    #[cfg(feature = "webgpu")]
    dev: Option<GpuDevice>,
}

impl GpuGnnEmbed {
    pub fn new() -> Option<Self> {
        Some(Self {
            #[cfg(feature = "webgpu")]
            dev: GpuDevice::new(),
        })
    }

    /// Compute a `EMBED_DIM`-dimensional embedding for `node_id`.
    ///
    /// Returns zeros until the WGSL GraphSAGE kernel is implemented.
    pub fn embed(&self, _node_id: u32) -> Vec<f32> {
        #[cfg(feature = "webgpu")]
        if let Some(_dev) = &self.dev {
            // TODO: dispatch WGSL_GNN_EMBED kernel
            // 1. Upload node feature buffer + adjacency
            // 2. dev.dispatch(&pipeline, &bind_group, n_groups)
            // 3. Readback f32 embedding buffer
        }
        vec![0.0f32; EMBED_DIM]
    }

    /// Batch embed all nodes 0..n_nodes.
    pub fn embed_all(&self, n_nodes: usize) -> Vec<Vec<f32>> {
        (0..n_nodes as u32).map(|id| self.embed(id)).collect()
    }

    pub fn embed_dim(&self) -> usize { EMBED_DIM }
}
