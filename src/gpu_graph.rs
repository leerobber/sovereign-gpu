/// GPU graph engine — CSR layout + WGSL kernel dispatch.
///
/// One `GpuGraph` per graph instance. Algorithms are dispatched to
/// the WebGPU device and results read back synchronously.

#[cfg(feature = "webgpu")]
use crate::backends::webgpu::GpuDevice;

const WGSL_BFS:      &str = include_str!("kernels/bfs.wgsl");
const WGSL_DFS:      &str = include_str!("kernels/dfs.wgsl");
const WGSL_SHORTEST: &str = include_str!("kernels/shortest.wgsl");
const WGSL_TOPO:     &str = include_str!("kernels/topo.wgsl");

const WORKGROUP: u32 = 64;

// ── CSR graph ─────────────────────────────────────────────────────────────────

pub struct CsrGraph {
    /// row_ptr[i] .. row_ptr[i+1] = neighbours of node i.
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub n_nodes: usize,
}

impl CsrGraph {
    /// Build a CSR graph from an edge list (0-indexed node IDs).
    pub fn from_edges(n_nodes: usize, edges: &[(u32, u32)]) -> Self {
        let mut row_ptr = vec![0u32; n_nodes + 1];
        for &(u, _v) in edges {
            if (u as usize) < n_nodes {
                row_ptr[u as usize + 1] += 1;
            }
        }
        for i in 0..n_nodes {
            row_ptr[i + 1] += row_ptr[i];
        }
        let mut col_idx = vec![0u32; edges.len()];
        let mut pos     = row_ptr[..n_nodes].to_vec();
        for &(u, v) in edges {
            if (u as usize) < n_nodes {
                let idx = pos[u as usize] as usize;
                col_idx[idx] = v;
                pos[u as usize] += 1;
            }
        }
        Self { row_ptr, col_idx, n_nodes }
    }
}

// ── GPU graph engine ──────────────────────────────────────────────────────────

pub struct GpuGraph {
    graph: CsrGraph,
    #[cfg(feature = "webgpu")]
    dev: Option<GpuDevice>,
}

impl GpuGraph {
    /// Empty graph — used internally by UnifiedGpu; populate via load().
    pub fn empty() -> Self {
        Self::new(0, vec![])
    }

    /// Replace the graph data (preserves the GPU device).
    pub fn load(&mut self, n_nodes: usize, edges: Vec<(u32, u32)>) {
        self.graph = CsrGraph::from_edges(n_nodes, &edges);
    }

    pub fn new(n_nodes: usize, edges: Vec<(u32, u32)>) -> Self {
        Self {
            graph: CsrGraph::from_edges(n_nodes, &edges),
            #[cfg(feature = "webgpu")]
            dev: GpuDevice::new(),
        }
    }

    #[cfg(feature = "webgpu")]
    fn dev(&self) -> &GpuDevice {
        self.dev.as_ref().expect("no GPU device available")
    }

    // ── BFS ───────────────────────────────────────────────────────────────────

    pub fn bfs(&self, start: u32) -> Vec<u32> {
        #[cfg(feature = "webgpu")]
        {
            if self.dev.is_some() {
                return self.bfs_gpu(start);
            }
        }
        self.bfs_cpu(start)
    }

    fn bfs_cpu(&self, start: u32) -> Vec<u32> {
        let n = self.graph.n_nodes;
        let mut visited = vec![false; n];
        let mut order   = Vec::new();
        if start as usize >= n { return order; }
        visited[start as usize] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let s = self.graph.row_ptr[node as usize] as usize;
            let e = self.graph.row_ptr[node as usize + 1] as usize;
            for &nbr in &self.graph.col_idx[s..e] {
                if !visited[nbr as usize] {
                    visited[nbr as usize] = true;
                    queue.push_back(nbr);
                }
            }
        }
        order
    }

    #[cfg(feature = "webgpu")]
    fn bfs_gpu(&self, start: u32) -> Vec<u32> {
        let dev = self.dev();
        let n   = self.graph.n_nodes;
        let pipeline = dev.compile(WGSL_BFS, "main");

        let b_row_ptr  = dev.upload_u32(&self.graph.row_ptr, "row_ptr");
        let b_col_idx  = dev.upload_u32(&self.graph.col_idx, "col_idx");

        let mut visited_data = vec![0u32; n];
        if (start as usize) < n { visited_data[start as usize] = 1; }
        let b_visited  = dev.upload_u32(&visited_data, "visited");

        // frontier_in: [count, node0, node1, ...] — slot 0 = count
        let mut fin_data = vec![0u32; n + 1];
        fin_data[0] = 1;
        fin_data[1] = start;
        let b_fin      = dev.upload_u32(&fin_data, "frontier_in");
        let b_fout     = dev.alloc_u32(n + 1, "frontier_out");
        let b_fout_len = dev.upload_u32(&[0u32], "frontier_out_len");

        let layout = pipeline.get_bind_group_layout(0);
        let mut order = vec![start];
        let mut frontier_len = 1u32;

        // BFS levels
        for _level in 0..n {
            if frontier_len == 0 { break; }

            // Clear fout_len
            dev.queue.write_buffer(&b_fout_len, 0, bytemuck::bytes_of(&0u32));

            let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b_row_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b_col_idx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_visited.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: b_fin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: b_fout.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: b_fout_len.as_entire_binding() },
                ],
            });

            let groups = (frontier_len + WORKGROUP - 1) / WORKGROUP;
            dev.dispatch(&pipeline, &bg, groups);

            // Read back new frontier length
            let fout_len_data = dev.readback_u32(&b_fout_len, 1);
            frontier_len = fout_len_data[0];
            if frontier_len == 0 { break; }

            // Read back new frontier nodes
            let fout_data = dev.readback_u32(&b_fout, frontier_len as usize + 1);
            order.extend_from_slice(&fout_data[1..frontier_len as usize + 1]);

            // Swap: copy fout → fin, reset fout
            let mut new_fin = vec![0u32; n + 1];
            new_fin[0] = frontier_len;
            new_fin[1..frontier_len as usize + 1]
                .copy_from_slice(&fout_data[1..frontier_len as usize + 1]);
            dev.queue.write_buffer(&b_fin, 0, bytemuck::cast_slice(&new_fin));
        }
        order
    }

    // ── DFS (label propagation, gives reachability order) ─────────────────────

    pub fn dfs(&self, start: u32) -> Vec<u32> {
        // CPU DFS — GPU label-prop DFS gives reachability, not exact DFS order.
        let n = self.graph.n_nodes;
        let mut visited = vec![false; n];
        let mut order   = Vec::new();
        if start as usize >= n { return order; }
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node as usize] { continue; }
            visited[node as usize] = true;
            order.push(node);
            let s = self.graph.row_ptr[node as usize] as usize;
            let e = self.graph.row_ptr[node as usize + 1] as usize;
            for &nbr in self.graph.col_idx[s..e].iter().rev() {
                if !visited[nbr as usize] { stack.push(nbr); }
            }
        }
        order
    }

    // ── Shortest paths ────────────────────────────────────────────────────────

    pub fn shortest_paths(&self, start: u32) -> Vec<(u32, u32)> {
        #[cfg(feature = "webgpu")]
        {
            if self.dev.is_some() {
                return self.shortest_gpu(start);
            }
        }
        self.shortest_cpu(start)
    }

    fn shortest_cpu(&self, start: u32) -> Vec<(u32, u32)> {
        let n = self.graph.n_nodes;
        let mut dist = vec![u32::MAX; n];
        if (start as usize) < n { dist[start as usize] = 0; }
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let d = dist[node as usize];
            let s = self.graph.row_ptr[node as usize] as usize;
            let e = self.graph.row_ptr[node as usize + 1] as usize;
            for &nbr in &self.graph.col_idx[s..e] {
                if dist[nbr as usize] == u32::MAX {
                    dist[nbr as usize] = d + 1;
                    queue.push_back(nbr);
                }
            }
        }
        dist.iter().enumerate()
            .filter(|&(_, &d)| d != u32::MAX)
            .map(|(i, &d)| (i as u32, d))
            .collect()
    }

    #[cfg(feature = "webgpu")]
    fn shortest_gpu(&self, start: u32) -> Vec<(u32, u32)> {
        let dev  = self.dev();
        let n    = self.graph.n_nodes;
        let pipeline = dev.compile(WGSL_SHORTEST, "main");

        let b_row_ptr = dev.upload_u32(&self.graph.row_ptr, "row_ptr");
        let b_col_idx = dev.upload_u32(&self.graph.col_idx, "col_idx");

        let mut dist_data = vec![u32::MAX; n];
        if (start as usize) < n { dist_data[start as usize] = 0; }
        let b_dist    = dev.upload_u32(&dist_data, "dist");
        let b_changed = dev.upload_u32(&[0u32], "changed");

        let layout = pipeline.get_bind_group_layout(0);
        let groups = ((n as u32) + WORKGROUP - 1) / WORKGROUP;

        for _pass in 0..n {
            dev.queue.write_buffer(&b_changed, 0, bytemuck::bytes_of(&0u32));
            let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b_row_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b_col_idx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_dist.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: b_changed.as_entire_binding() },
                ],
            });
            dev.dispatch(&pipeline, &bg, groups);
            let ch = dev.readback_u32(&b_changed, 1);
            if ch[0] == 0 { break; }
        }

        let dist_out = dev.readback_u32(&b_dist, n);
        dist_out.iter().enumerate()
            .filter(|&(_, &d)| d != u32::MAX)
            .map(|(i, &d)| (i as u32, d))
            .collect()
    }

    // ── Topo sort ─────────────────────────────────────────────────────────────

    pub fn topo_sort(&self) -> Option<Vec<u32>> {
        // CPU Kahn — GPU version used for large graphs when available.
        let n = self.graph.n_nodes;
        let mut in_deg = vec![0u32; n];
        for &v in &self.graph.col_idx { in_deg[v as usize] += 1; }
        let mut queue: std::collections::VecDeque<u32> = (0..n as u32)
            .filter(|&i| in_deg[i as usize] == 0)
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let s = self.graph.row_ptr[node as usize] as usize;
            let e = self.graph.row_ptr[node as usize + 1] as usize;
            for &nbr in &self.graph.col_idx[s..e] {
                in_deg[nbr as usize] -= 1;
                if in_deg[nbr as usize] == 0 { queue.push_back(nbr); }
            }
        }
        if order.len() == n { Some(order) } else { None }
    }
}
