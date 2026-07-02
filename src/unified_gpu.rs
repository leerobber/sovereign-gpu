/// UnifiedGpu — single Python-facing class that dispatches to WebGPU, CUDA, or Vulkan.
use pyo3::prelude::*;

use crate::backends::cuda::CudaGraph;
use crate::backends::vulkan::VkGraph;
use crate::gnn::GpuGnnEmbed;
use crate::gpu_graph::GpuGraph;
use crate::planner::GpuPlanner;

#[pyclass(name = "UnifiedGpu")]
pub struct UnifiedGpu {
    backend:      String,
    webgpu_graph: Option<GpuGraph>,
    cuda_graph:   Option<CudaGraph>,
    vulkan_graph: Option<VkGraph>,
    gnn:          Option<GpuGnnEmbed>,
    planner:      Option<GpuPlanner>,
}

#[pymethods]
impl UnifiedGpu {
    /// Create a UnifiedGpu.
    ///
    /// Args:
    ///     backend: "webgpu" (default) | "cuda" | "vulkan"
    ///              CUDA and Vulkan currently use stub implementations;
    ///              enable their Cargo features + supply real kernels to activate.
    #[new]
    #[pyo3(signature = (backend=None))]
    pub fn new(backend: Option<String>) -> Self {
        let backend = backend.unwrap_or_else(|| "webgpu".to_string());
        Self {
            webgpu_graph: Some(GpuGraph::empty()),
            cuda_graph:   CudaGraph::new(0, vec![]).ok(),
            vulkan_graph: VkGraph::new(0, vec![]).ok(),
            gnn:          GpuGnnEmbed::new(32, 128),
            planner:      GpuPlanner::new(),
            backend,
        }
    }

    /// Load (or replace) the graph used for BFS/DFS/shortest/topo dispatch.
    ///
    /// Args:
    ///     n_nodes: number of nodes (0-indexed)
    ///     edges:   list of (from_u32, to_u32) tuples
    pub fn load_graph(&mut self, n_nodes: usize, edges: Vec<(u32, u32)>) {
        if let Some(ref mut g) = self.webgpu_graph {
            g.load(n_nodes, edges.clone());
        }
        if let Some(ref mut g) = self.cuda_graph {
            g.load(n_nodes, edges.clone());
        }
        if let Some(ref mut g) = self.vulkan_graph {
            g.load(n_nodes, edges);
        }
    }

    /// BFS from `start`; returns visited node IDs in level order.
    pub fn bfs(&self, start: u32) -> Vec<u32> {
        match self.backend.as_str() {
            "cuda"   => self.cuda_graph.as_ref().map(|g| g.bfs(start)).unwrap_or_default(),
            "vulkan" => self.vulkan_graph.as_ref().map(|g| g.bfs(start)).unwrap_or_default(),
            _        => self.webgpu_graph.as_ref().map(|g| g.bfs(start)).unwrap_or_default(),
        }
    }

    /// DFS from `start`; returns visited node IDs in DFS pre-order.
    pub fn dfs(&self, start: u32) -> Vec<u32> {
        match self.backend.as_str() {
            "cuda"   => self.cuda_graph.as_ref().map(|g| g.dfs(start)).unwrap_or_default(),
            "vulkan" => self.vulkan_graph.as_ref().map(|g| g.dfs(start)).unwrap_or_default(),
            _        => self.webgpu_graph.as_ref().map(|g| g.dfs(start)).unwrap_or_default(),
        }
    }

    /// Shortest-path distances from `start`; returns (node_id, dist) pairs.
    pub fn shortest_paths(&self, start: u32) -> Vec<(u32, u32)> {
        match self.backend.as_str() {
            "cuda" => self.cuda_graph.as_ref()
                .map(|g| g.shortest_paths(start)).unwrap_or_default(),
            "vulkan" => self.vulkan_graph.as_ref()
                .map(|g| g.shortest_paths(start)).unwrap_or_default(),
            _ => self.webgpu_graph.as_ref()
                .map(|g| g.shortest_paths(start)).unwrap_or_default(),
        }
    }

    /// Topological sort; returns None if the graph has a cycle.
    pub fn topo_sort(&self) -> Option<Vec<u32>> {
        match self.backend.as_str() {
            "cuda"   => self.cuda_graph.as_ref().and_then(|g| g.topo_sort()),
            "vulkan" => self.vulkan_graph.as_ref().and_then(|g| g.topo_sort()),
            _        => self.webgpu_graph.as_ref().and_then(|g| g.topo_sort()),
        }
    }

    /// GraphSAGE-Mean embedding for all nodes in one GPU dispatch.
    ///
    /// Args:
    ///     row_ptr:    CSR row pointers (len = n_nodes + 1)
    ///     col_idx:    CSR column indices (len = n_edges)
    ///     node_feats: node feature matrix, row-major (n_nodes × feat_dim)
    ///     weight:     projection matrix, row-major (feat_dim × embed_dim)
    ///
    /// Returns flattened embeddings (n_nodes × embed_dim).
    pub fn embed(
        &self,
        row_ptr:    Vec<u32>,
        col_idx:    Vec<u32>,
        node_feats: Vec<f32>,
        weight:     Vec<f32>,
    ) -> Vec<f32> {
        self.gnn
            .as_ref()
            .map(|g| g.embed(&row_ptr, &col_idx, &node_feats, &weight))
            .unwrap_or_default()
    }

    /// Run the GPU planner pipeline on serialised block bytes.
    /// Returns refined output bytes (pass-through until WGSL kernels are filled in).
    pub fn plan(&self, block_bytes: &[u8]) -> Vec<u8> {
        self.planner.as_ref().map(|p| p.run(block_bytes)).unwrap_or_default()
    }

    /// Whether the planner has a GPU device ready.
    pub fn planner_ready(&self) -> bool {
        self.planner.as_ref().map(|p| p.is_gpu_ready()).unwrap_or(false)
    }

    /// Active backend name.
    #[getter]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Switch backend at runtime (e.g. "webgpu" → "cuda").
    #[setter]
    pub fn set_backend(&mut self, backend: String) {
        self.backend = backend;
    }

    pub fn __repr__(&self) -> String {
        format!("UnifiedGpu(backend='{}')", self.backend)
    }
}
