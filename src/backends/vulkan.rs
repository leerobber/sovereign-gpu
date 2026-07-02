/// Vulkan compute backend stub.
///
/// To activate real Vulkan compute kernels, enable the "vulkan" feature and
/// replace this stub with `ash`-backed SPIR-V dispatch.
/// Current build produces no-ops so the crate compiles without a Vulkan SDK.

pub struct VkGraph {
    n_nodes: usize,
    edges:   Vec<(u32, u32)>,
}

impl VkGraph {
    pub fn new(n_nodes: usize, edges: Vec<(u32, u32)>) -> Result<Self, String> {
        Ok(Self { n_nodes, edges })
    }

    pub fn load(&mut self, n_nodes: usize, edges: Vec<(u32, u32)>) {
        self.n_nodes = n_nodes;
        self.edges   = edges;
    }

    pub fn node_count(&self) -> usize { self.n_nodes }

    /// BFS — stub returns empty until Vulkan SPIR-V is wired.
    pub fn bfs(&self, _start: u32) -> Vec<u32> {
        // TODO: vkCreateComputePipeline + vkCmdDispatch with BFS SPIR-V
        Vec::new()
    }

    pub fn dfs(&self, _start: u32) -> Vec<u32> {
        // TODO: Vulkan DFS compute dispatch
        Vec::new()
    }

    pub fn shortest_paths(&self, _start: u32) -> Vec<(u32, u32)> {
        // TODO: Vulkan Bellman-Ford dispatch
        Vec::new()
    }

    pub fn topo_sort(&self) -> Option<Vec<u32>> {
        // TODO: Vulkan Kahn's dispatch
        None
    }
}

// ── Real Vulkan implementation (feature = "vulkan") ───────────────────────────
// Uncomment and fill in when the Vulkan SDK is available:
//
// #[cfg(feature = "vulkan")]
// mod real {
//     use ash::{vk, Instance, Device};
//
//     pub struct VkDevice { pub instance: Instance, pub device: Device, pub queue: vk::Queue }
//
//     impl VkDevice {
//         pub fn new() -> anyhow::Result<Self> {
//             // create instance, pick physical device, create logical device + compute queue
//             unimplemented!()
//         }
//     }
// }
