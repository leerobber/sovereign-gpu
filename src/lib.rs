mod backends;
mod gnn;
mod gpu_graph;
mod planner;
mod py_gpu_graph;
mod unified_gpu;

use pyo3::prelude::*;

#[pymodule]
fn sovereign_gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_gpu_graph::PyGpuGraph>()?;
    m.add_class::<unified_gpu::UnifiedGpu>()?;
    Ok(())
}
