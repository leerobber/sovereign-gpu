/// CUDA backend stub.
///
/// To activate real CUDA kernels, enable the "cuda" feature in Cargo.toml
/// and replace this stub with `cust`-backed implementations.
/// The current build produces no-ops so the crate compiles without a CUDA toolkit.

pub struct CudaGraph {
    n_nodes: usize,
    edges:   Vec<(u32, u32)>,
}

impl CudaGraph {
    pub fn new(n_nodes: usize, edges: Vec<(u32, u32)>) -> Result<Self, String> {
        Ok(Self { n_nodes, edges })
    }

    pub fn load(&mut self, n_nodes: usize, edges: Vec<(u32, u32)>) {
        self.n_nodes = n_nodes;
        self.edges   = edges;
    }

    pub fn node_count(&self) -> usize { self.n_nodes }

    /// BFS — stub returns empty until CUDA kernel is wired.
    pub fn bfs(&self, _start: u32) -> Vec<u32> {
        // TODO: cust::function! launch cuda_bfs<<<blocks, threads>>> (row_ptr, col_idx, visited, frontier, start)
        Vec::new()
    }

    pub fn dfs(&self, _start: u32) -> Vec<u32> {
        // TODO: CUDA DFS label-propagation kernel
        Vec::new()
    }

    pub fn shortest_paths(&self, _start: u32) -> Vec<(u32, u32)> {
        // TODO: CUDA Bellman-Ford kernel
        Vec::new()
    }

    pub fn topo_sort(&self) -> Option<Vec<u32>> {
        // TODO: CUDA Kahn's kernel
        None
    }
}

// ── Real CUDA implementation (feature = "cuda") ───────────────────────────────
// Uncomment and fill in when the CUDA toolkit is available:
//
// #[cfg(feature = "cuda")]
// mod real {
//     use cust::prelude::*;
//
//     pub struct CudaDevice { pub ctx: Context, pub stream: Stream }
//
//     impl CudaDevice {
//         pub fn new() -> cust::error::CudaResult<Self> {
//             cust::init(cust::CudaFlags::empty())?;
//             let device = Device::get_device(0)?;
//             let ctx = Context::create_and_push(
//                 ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//             let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
//             Ok(Self { ctx, stream })
//         }
//     }
// }
