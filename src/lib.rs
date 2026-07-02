mod backends;
mod gpu_graph;
mod py_gpu_graph;

use pyo3::prelude::*;

#[pymodule]
fn sovereign_gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_gpu_graph::PyGpuGraph>()?;
    Ok(())
}
