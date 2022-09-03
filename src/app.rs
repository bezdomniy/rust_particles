use rand::{thread_rng, Rng};
use std::borrow::Cow;
use wgpu::{
    util::DeviceExt, Adapter, Buffer, Device, Instance, Queue, RenderPipeline, Surface,
    SurfaceConfiguration,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const NUM_PARTICLES: u32 = 1024;
const PARTICLE_SIZE: f32 = 5f32;
// const PARTICLES_PER_GROUP: u32 = 64;

pub struct App {
    event_loop: EventLoop<()>,
    instance: Instance,
    surface: Surface,
    window: Window,
    adapter: Option<Adapter>,
    device: Option<Device>,
    queue: Option<Queue>,
    render_pipeline: Option<RenderPipeline>,
    config: Option<SurfaceConfiguration>,
    particle_buffer: Option<Buffer>,
    vertices_buffer: Option<Buffer>,
}

impl App {
    pub fn new() -> Self {
        let event_loop = EventLoop::new();
        let window = winit::window::Window::new(&event_loop).unwrap();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(&window) };

        App {
            event_loop,
            instance,
            surface,
            window,
            adapter: None,
            config: None,
            device: None,
            queue: None,
            render_pipeline: None,
            particle_buffer: None,
            vertices_buffer: None,
        }
    }

    pub async fn setup(&mut self) {
        let size = self.window.inner_size();
        let adapter = self
            .instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&self.surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/particles.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let swapchain_format = self.surface.get_supported_formats(&adapter)[0];

        // buffer for the three 2d triangle vertices of each instance
        let mut vertex_buffer_data = [
            0.001f32, 0.001f32, -0.001f32, -0.001f32, 0.001f32, -0.001f32, 0.001f32, 0.001f32,
            -0.001f32, -0.001f32, -0.001f32, 0.001f32,
        ];

        for v in vertex_buffer_data.iter_mut() {
            *v *= PARTICLE_SIZE
        }

        let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let mut initial_particle_data = vec![0f32; (2 * NUM_PARTICLES) as usize];
        let unif = || thread_rng().gen_range(-1f32..=1f32); // Generate a num (-1, 1)
                                                            // let disc = || thread_rng().gen_range(0u32..4u32);
        for particle_instance_chunk in initial_particle_data.chunks_mut(2) {
            particle_instance_chunk[0] = unif(); // posx
            particle_instance_chunk[1] = unif(); // posy
        }

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&initial_particle_data),
            usage: wgpu::BufferUsages::VERTEX
                // | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        // // calculates number of work groups from PARTICLES_PER_GROUP constant
        // let work_group_count =
        //     ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "main_vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "main_fs",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.config = Some(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        });

        self.surface
            .configure(&device, &self.config.as_ref().unwrap());

        self.device = Some(device);
        self.queue = Some(queue);
        self.adapter = Some(adapter);
        self.render_pipeline = Some(render_pipeline);
        self.particle_buffer = Some(particle_buffer);
        self.vertices_buffer = Some(vertices_buffer);
    }

    pub async fn run(mut self) {
        let mut config = self.config.take().unwrap();
        let device = self.device.take();
        let queue = self.queue.take();
        let render_pipeline = self.render_pipeline.take();
        let particle_buffer = self.particle_buffer.take();
        let vertices_buffer = self.vertices_buffer.take();

        self.event_loop.run(move |event, _, control_flow| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&self);
            let _ = (&self.instance, &self.adapter, &render_pipeline);

            *control_flow = ControlFlow::Wait;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    // Reconfigure the surface with the new size
                    config.width = size.width;
                    config.height = size.height;
                    self.surface.configure(&device.as_ref().unwrap(), &config);
                    // On macos the window needs to be redrawn manually after resizing
                    self.window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let frame = self
                        .surface
                        .get_current_texture()
                        .expect("Failed to acquire next swap chain texture");
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device
                        .as_ref()
                        .unwrap()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: true,
                                },
                            })],
                            depth_stencil_attachment: None,
                        });
                        rpass.set_pipeline(&render_pipeline.as_ref().unwrap());
                        rpass.set_vertex_buffer(0, particle_buffer.as_ref().unwrap().slice(..));
                        rpass.set_vertex_buffer(1, vertices_buffer.as_ref().unwrap().slice(..));
                        rpass.draw(0..6, 0..NUM_PARTICLES);
                    }

                    queue.as_ref().unwrap().submit(Some(encoder.finish()));
                    frame.present();
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {}
            }
        });
    }
}
