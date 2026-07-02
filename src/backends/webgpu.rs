/// WebGPU backend — wraps wgpu device/queue and compiles WGSL shaders.
use wgpu::util::DeviceExt;

pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
}

impl GpuDevice {
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter  = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;
            Some(GpuDevice { device, queue })
        })
    }

    /// Upload a `u32` slice to a GPU storage buffer.
    pub fn upload_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(label),
            contents: bytemuck::cast_slice(data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create an uninitialised GPU storage buffer (n u32 elements).
    pub fn alloc_u32(&self, n: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(label),
            size:               (n * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Readback buffer used for GPU → CPU copy.
    pub fn readback_buf(&self, size_bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("readback"),
            size:               size_bytes,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Compile a WGSL shader and return a ComputePipeline.
    pub fn compile(&self, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some(entry),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some(entry),
            layout:      None,
            module:      &module,
            entry_point: entry,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    }

    /// Dispatch a compute pipeline with n_groups workgroups and wait.
    pub fn dispatch(
        &self,
        pipeline:   &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n_groups:   u32,
    ) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:                    Some("pass"),
                timestamp_writes:         None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Copy `src` → `dst` (must be sized <= src).
    pub fn copy_buf(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size_bytes: u64) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy"),
        });
        enc.copy_buffer_to_buffer(src, 0, dst, 0, size_bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Synchronously read a buffer back to a Vec<u32>.
    pub fn readback_u32(&self, src: &wgpu::Buffer, n: usize) -> Vec<u32> {
        let bytes = (n * 4) as u64;
        let rb    = self.readback_buf(bytes);
        self.copy_buf(src, &rb, bytes);
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let view   = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        rb.unmap();
        result
    }
}
