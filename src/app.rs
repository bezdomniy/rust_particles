use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{
    borrow::Cow,
    time::{Duration, Instant},
};
use wgpu::{
    util::DeviceExt, Adapter, Buffer, Device, Instance, Queue, RenderPipeline, Surface,
    SurfaceConfiguration,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const FRAMERATE: u32 = 30;
const NUM_PARTICLES: usize = 256;

const P_COLOUR: [f32; 4] = [0.5, 0.0, 0.5, 0.0];

const PARTICLE_SIZE: f32 = 3f32;
// const PARTICLES_PER_GROUP: u32 = 64;

// const rng: ThreadRng = thread_rng();
// const unif = || thread_rng().gen_range(-1f32..=1f32); // Generate a num (-1, 1)

pub struct App {
    event_loop: EventLoop<()>,
    instance: Instance,
    surface: Surface,
    window: Window,
    particle_data: Vec<Particle>,
    particle_offsets: [usize; 3],
    adapter: Option<Adapter>,
    device: Option<Device>,
    queue: Option<Queue>,
    render_pipeline: Option<RenderPipeline>,
    config: Option<SurfaceConfiguration>,
    particle_buffer: Option<Buffer>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    posx: f32,
    posy: f32,
    velx: f32,
    vely: f32,
    cls: u32,
}

fn init_particles(particle_data: &mut Vec<Particle>, particle_offsets: &[usize]) {
    for _ in 0..particle_offsets[0] {
        particle_data.push(Particle {
            posx: thread_rng().gen_range(-1f32..=1f32),
            posy: thread_rng().gen_range(-1f32..=1f32),
            velx: thread_rng().gen_range(-1f32..=1f32),
            vely: thread_rng().gen_range(-1f32..=1f32),
            cls: 0,
        })
    }
    for _ in particle_offsets[0]..particle_offsets[1] {
        particle_data.push(Particle {
            posx: thread_rng().gen_range(-1f32..=1f32),
            posy: thread_rng().gen_range(-1f32..=1f32),
            velx: thread_rng().gen_range(-1f32..=1f32),
            vely: thread_rng().gen_range(-1f32..=1f32),
            cls: 1,
        })
    }
    for _ in particle_offsets[1]..particle_offsets[2] {
        particle_data.push(Particle {
            posx: thread_rng().gen_range(-1f32..=1f32),
            posy: thread_rng().gen_range(-1f32..=1f32),
            velx: thread_rng().gen_range(-1f32..=1f32),
            vely: thread_rng().gen_range(-1f32..=1f32),
            cls: 2,
        })
    }
    for _ in particle_offsets[2]..NUM_PARTICLES {
        particle_data.push(Particle {
            posx: thread_rng().gen_range(-1f32..=1f32),
            posy: thread_rng().gen_range(-1f32..=1f32),
            velx: thread_rng().gen_range(-1f32..=1f32),
            vely: thread_rng().gen_range(-1f32..=1f32),
            cls: 3,
        })
    }
}

// // Interaction between 2 particle groups
// fn interaction(
//     particle_data: &mut [Particle],
//     particle_offsets: &[usize],
//     group1_idx: usize,
//     group2_idx: usize,
//     g: f32,
//     radius: f32,
// ) {
//     let group1_start = if group1_idx == 0 {
//         0
//     } else {
//         particle_offsets[group1_idx - 1]
//     };

//     let group2_start = if group2_idx == 0 {
//         0
//     } else {
//         particle_offsets[group2_idx - 1]
//     };

//     let group1_slice = &mut particle_data[group1_start..particle_offsets[group1_idx] as usize];
//     let group2_slice = &particle_data[group2_start..particle_offsets[group2_idx] as usize];

//     let g = g / -100f32;

//     // omp_set_num_threads(4);
//     // #pragma omp parallel for
//     for p1 in group1_slice.iter_mut() {
//         let mut fx = 0f32;
//         let mut fy = 0f32;
//         for p2 in group2_slice.iter() {
//             let dx = p1.posx - p2.posx;
//             let dy = p1.posy - p2.posy;
//             let r = (dx * dx + dy * dy).sqrt();
//             if r < radius && r > 0f32 {
//                 fx += dx / r;
//                 fy += dy / r;
//             }
//         }

//         p1.velx = (p1.velx + (fx * g)) * 0.5;
//         p1.vely = (p1.vely + (fy * g)) * 0.5;
//         p1.posx += p1.velx;
//         p1.posy += p1.vely;

//         // if (boundsToggle) {
//         //     //not good enough! Need fixing
//         //     if (p1.posx >= (1920 - 10) as f32) || (p1.posx <= (550 + 10) as f32) {
//         //         p1.velx *= -1f32;
//         //     }
//         //     if (p1.posy >= (1024 - 10) as f32) || (p1.posy <= (0 + 10) as f32) {
//         //         p1.vely *= -1f32;
//         //     }
//         // }

//         // (*Group1)[i] = p1;
//     }
// }

impl App {
    pub fn new() -> Self {
        let event_loop = EventLoop::new();
        let window = winit::window::Window::new(&event_loop).unwrap();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(&window) };

        let particle_offsets = [
            (NUM_PARTICLES as f32 * P_COLOUR[0]) as usize,
            (NUM_PARTICLES as f32 * P_COLOUR[0]) as usize
                + (NUM_PARTICLES as f32 * P_COLOUR[1]) as usize,
            (NUM_PARTICLES as f32 * P_COLOUR[0]) as usize
                + (NUM_PARTICLES as f32 * P_COLOUR[1]) as usize
                + (NUM_PARTICLES as f32 * P_COLOUR[2]) as usize,
        ];

        App {
            event_loop,
            instance,
            surface,
            window,
            particle_offsets,
            particle_data: Vec::with_capacity(NUM_PARTICLES),
            adapter: None,
            config: None,
            device: None,
            queue: None,
            render_pipeline: None,
            particle_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
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
            0.001f32, 0.001f32, -0.001f32, -0.001f32, 0.001f32, -0.001f32, -0.001f32, 0.001f32,
        ];

        for v in vertex_buffer_data.iter_mut() {
            *v *= PARTICLE_SIZE
        }

        let index_buffer_data = [0u16, 1u16, 2u16, 0u16, 1u16, 3u16];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&index_buffer_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        init_particles(&mut self.particle_data, &self.particle_offsets);

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&self.particle_data),
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
                        array_stride: 5 * 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Uint32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x2],
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
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
    }

    fn update(
        particle_data: &mut Vec<Particle>,
        particle_offsets: &[usize],
        buffer: &mut Buffer,
        queue: &mut Queue,
    ) {
        // let red_slice = &mut particle_data[0..particle_offsets[0] as usize];
        // let green_slice = &mut particle_data[particle_offsets[0]..particle_offsets[1] as usize];
        // let blue_slice = &mut particle_data[particle_offsets[1]..particle_offsets[2] as usize];
        // let white_slice = &mut particle_data[particle_offsets[2]..particle_offsets[3] as usize];

        let powerSliderGG = 10f32;
        let powerSliderGR = 10f32;
        let powerSliderGW = 10f32;
        let powerSliderGB = 10f32;

        let powerSliderRG = 10f32;
        let powerSliderRR = 10f32;
        let powerSliderRW = 10f32;
        let powerSliderRB = 10f32;

        let powerSliderWG = 10f32;
        let powerSliderWR = 10f32;
        let powerSliderWW = 10f32;
        let powerSliderWB = 10f32;

        let powerSliderBG = 10f32;
        let powerSliderBR = 10f32;
        let powerSliderBW = 10f32;
        let powerSliderBB = 10f32;

        let vSliderGG = 0f32;
        let vSliderGR = 0f32;
        let vSliderGW = 0f32;
        let vSliderGB = 0f32;

        let vSliderRG = 0f32;
        let vSliderRR = 0f32;
        let vSliderRW = 0f32;
        let vSliderRB = 0f32;

        let vSliderBG = 0f32;
        let vSliderBR = 0f32;
        let vSliderBW = 0f32;
        let vSliderBB = 0f32;

        let vSliderWG = 0f32;
        let vSliderWR = 0f32;
        let vSliderWW = 0f32;
        let vSliderWB = 0f32;

        for i in 0..4 {
            for j in 0..4 {
                // interaction(
                //     particle_data.as_mut_slice(),
                //     particle_offsets,
                //     i,
                //     j,
                //     powerSliderGG,
                //     vSliderGG,
                // );
            }
        }

        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&particle_data));
    }

    pub async fn run(mut self) {
        let mut config = self.config.take().unwrap();
        let device = self.device.take();
        let mut queue = self.queue.take();
        let render_pipeline = self.render_pipeline.take();
        let mut particle_buffer = self.particle_buffer.take();
        let vertices_buffer = self.vertex_buffer.take();

        let mut last_update_inst = Instant::now();

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
                Event::RedrawEventsCleared => {
                    let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE as f64);
                    let time_since_last_frame = last_update_inst.elapsed();

                    if time_since_last_frame >= target_frametime {
                        last_update_inst = Instant::now();
                        log::info!(
                            "Framerate: {}",
                            1000u128 / time_since_last_frame.as_millis()
                        );

                        Self::update(
                            &mut self.particle_data,
                            &self.particle_offsets,
                            particle_buffer.as_mut().unwrap(),
                            queue.as_mut().unwrap(),
                        );
                    } else {
                        // exit(0);
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }
                }
                Event::MainEventsCleared => {
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
                        rpass.set_index_buffer(
                            self.index_buffer.as_ref().unwrap().slice(..),
                            wgpu::IndexFormat::Uint16,
                        );
                        rpass.set_vertex_buffer(0, particle_buffer.as_ref().unwrap().slice(..));
                        rpass.set_vertex_buffer(1, vertices_buffer.as_ref().unwrap().slice(..));
                        // rpass.draw(0..6, 0..NUM_PARTICLES);
                        rpass.draw_indexed(0..6 as u32, 0, 0..NUM_PARTICLES as u32);
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
