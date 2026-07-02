/// Python-facing GpuGraph wrapper.
use pyo3::prelude::*;
use crate::gpu_graph::GpuGraph;

#[pyclass(name = "GpuGraph")]
pub struct PyGpuGraph {
    inner: GpuGraph,
}

#[pymethods]
impl PyGpuGraph {
    /// Create a new GpuGraph.
    ///
    /// Args:
    ///     n_nodes: number of nodes (0-indexed)
    ///     edges:   list of (from_u32, to_u32) tuples
    #[new]
    pub fn new(n_nodes: usize, edges: Vec<(u32, u32)>) -> Self {
        Self { inner: GpuGraph::new(n_nodes, edges) }
    }

    /// BFS from `start`; returns visited node IDs in level order.
    pub fn bfs(&self, start: u32) -> Vec<u32> {
        self.inner.bfs(start)
    }

    /// DFS from `start`; returns visited node IDs in DFS pre-order.
    pub fn dfs(&self, start: u32) -> Vec<u32> {
        self.inner.dfs(start)
    }

    /// Shortest-path distances from `start`.
    /// Returns list of (node_id, distance) for all reachable nodes.
    pub fn shortest_paths(&self, start: u32) -> Vec<(u32, u32)> {
        self.inner.shortest_paths(start)
    }

    /// Topological sort; returns None if the graph has a cycle.
    pub fn topo_sort(&self) -> Option<Vec<u32>> {
        self.inner.topo_sort()
    }

    pub fn __repr__(&self) -> String {
        "GpuGraph".to_string()
    }
}
