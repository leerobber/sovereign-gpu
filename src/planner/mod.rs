/// GPU planner pipeline — expand → score → prune → refine.
///
/// Takes a serialised plan block (KernelBlock bytes) and runs it through
/// a 4-stage WGSL pipeline to produce a refined output block.
/// Current implementation is a pass-through stub pending WGSL kernel authoring.

#[cfg(feature = "webgpu")]
use crate::backends::webgpu::GpuDevice;

const _WGSL_EXPAND:  &str = include_str!("../kernels/planner_expand.wgsl");
const _WGSL_SCORE:   &str = include_str!("../kernels/planner_score.wgsl");
const _WGSL_PRUNE:   &str = include_str!("../kernels/planner_prune.wgsl");
const _WGSL_REFINE:  &str = include_str!("../kernels/planner_refine.wgsl");

pub struct GpuPlanner {
    #[cfg(feature = "webgpu")]
    dev: Option<GpuDevice>,
}

impl GpuPlanner {
    pub fn new() -> Option<Self> {
        Some(Self {
            #[cfg(feature = "webgpu")]
            dev: GpuDevice::new(),
        })
    }

    /// Run the planner pipeline on serialised block bytes.
    ///
    /// Returns refined output bytes. Currently a no-op stub — returns input unchanged.
    pub fn run(&self, block_bytes: &[u8]) -> Vec<u8> {
        #[cfg(feature = "webgpu")]
        if let Some(_dev) = &self.dev {
            // TODO: dispatch pipeline stages
            // Stage 1: expand   — generate candidate next states
            // Stage 2: score    — evaluate each candidate
            // Stage 3: prune    — keep top-k
            // Stage 4: refine   — polish the winning candidate
        }
        block_bytes.to_vec()   // pass-through until kernels are implemented
    }

    /// Whether a GPU device is available for planning.
    pub fn is_gpu_ready(&self) -> bool {
        #[cfg(feature = "webgpu")]
        { self.dev.is_some() }
        #[cfg(not(feature = "webgpu"))]
        { false }
    }
}
