use glam::{Mat4, UVec4, Vec2};
use rand::{distributions::Uniform, thread_rng, Rng};
use rayon::prelude::*;
use std::{
    borrow::Cow,
    time::{Duration, Instant},
};
use wgpu::{
    util::DeviceExt, Adapter, Buffer, Device, Instance, Queue, RenderPipeline, Surface,
    SurfaceConfiguration,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopWindowTarget},
    window::Window,
};

const FRAMERATE: u32 = 30;
const BOUNDS_TOGGLE: bool = true;
const PARTICLE_SIZE: f32 = 2f32;
// const PARTICLES_PER_GROUP: u32 = 64;

pub struct App {
    instance: Instance,
    surface: Surface,
    game_state: GameState,
    adapter: Option<Adapter>,
    device: Option<Device>,
    queue: Option<Queue>,
    render_pipeline: Option<RenderPipeline>,
    config: Option<SurfaceConfiguration>,
    particle_buffer: Option<Buffer>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
}

struct GameState {
    particle_data: Vec<Particle>,
    particle_offsets: [i32; 4],
    pub power_slider: Mat4,
    pub r_slider: Mat4,
    pub num_particles: UVec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
    cls: u32,
}

// Interaction between 2 particle groups
fn interaction(
    group1: &[Particle],
    group2: &[Particle],
    g: f32,
    radius: f32,
    viscosity: f32,
) -> Vec<Particle> {
    let g = g / 10000000f32;

    group1
        .par_iter()
        .map(|p1| {
            let mut f = Vec2::new(0f32, 0f32);
            group2.iter().for_each(|p2| {
                // let d_sq = p1.pos.distance_squared(p2.pos);

                // let force = if d_sq < radius * radius && d_sq > 0f32 {
                //     1f32 / d_sq.sqrt()
                // } else {
                //     0f32
                // };

                // f += (p1.pos - p2.pos) * force;

                let d = p1.pos - p2.pos;
                let r = d.length();
                if r < radius && r > 0f32 {
                    f += d.normalize() / r;
                }
            });

            if f.length() < f32::EPSILON {
                return *p1;
            }

            let mut vel = (p1.vel + (f * g)) * (1f32 - viscosity);
            let pos = p1.pos + vel;
            if BOUNDS_TOGGLE {
                //not good enough! Need fixing
                if (pos.x >= 1f32) || (pos.x <= -1f32) {
                    vel.x *= -1f32;
                }
                if (pos.y >= 1f32) || (pos.y <= -1f32) {
                    vel.y *= -1f32;
                }
            }

            Particle {
                pos,
                vel,
                cls: p1.cls,
            }
        })
        .collect()
}

impl App {
    pub fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(&window) };

        let mut rng = rand::thread_rng();

        let power_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(&Uniform::new(-40f32, 40f32)))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let r_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(&Uniform::new(0.01f32, 0.9f32)))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let num_particles = UVec4::new(3000, 3000, 3000, 3000);

        let game_state = GameState {
            particle_data: Vec::with_capacity(
                (num_particles.x + num_particles.y + num_particles.z + num_particles.w) as usize,
            ),
            particle_offsets: [
                if num_particles.x > 0 { 0 } else { -1 },
                if num_particles.y > 0 {
                    num_particles.x as i32
                } else {
                    -1
                },
                if num_particles.z > 0 {
                    (num_particles.x + num_particles.y) as i32
                } else {
                    -1
                },
                if num_particles.w > 0 {
                    (num_particles.x + num_particles.y + num_particles.z) as i32
                } else {
                    -1
                },
            ],
            power_slider: Mat4::from_cols_array(&power_vals),
            r_slider: Mat4::from_cols_array(&r_vals),
            // power_slider: Mat4::from_cols(
            //     Vec4::new(1f32, 1f32, -10f32, 10f32),
            //     Vec4::new(-20f32, 10f32, 10f32, 1f32),
            //     Vec4::new(10f32, 1f32, 10f32, 10f32),
            //     Vec4::new(1f32, -10f32, 10f32, 10f32),
            // ),

            // r_slider: Mat4::from_cols(
            //     Vec4::new(1f32, 1f32, 1f32, 1f32),
            //     Vec4::new(1f32, 1f32, 1f32, 1f32),
            //     Vec4::new(1f32, 1f32, 1f32, 1f32),
            //     Vec4::new(1f32, 1f32, 1f32, 1f32),
            // ),
            num_particles,
        };

        App {
            instance,
            surface,
            game_state,
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

    pub async fn setup(&mut self, physical_size: PhysicalSize<u32>) {
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

        self.init_particles();

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&self.game_state.particle_data),
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
            width: physical_size.width,
            height: physical_size.height,
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

    fn init_particles(&mut self) {
        for i in 0..4 {
            if self.game_state.particle_offsets[i as usize] < 0 {
                continue;
            }

            let start = self.game_state.particle_offsets[i as usize];
            let end = self
                .game_state
                .particle_offsets
                .into_iter()
                .skip(i + 1)
                .find(|&item| item > 0)
                .unwrap_or(
                    (self.game_state.num_particles.x
                        + self.game_state.num_particles.y
                        + self.game_state.num_particles.z
                        + self.game_state.num_particles.w) as i32,
                );

            for _ in start..end {
                self.game_state.particle_data.push(Particle {
                    pos: Vec2::new(
                        thread_rng().gen_range(-1f32..=1f32),
                        thread_rng().gen_range(-1f32..=1f32),
                    ),
                    vel: Vec2::new(
                        thread_rng().gen_range(-0.005f32..=0.005f32),
                        thread_rng().gen_range(-0.005f32..=0.005f32),
                    ),
                    cls: i as u32,
                })
            }
        }
    }

    fn update(&mut self) {
        for (i, group1_start) in self.game_state.particle_offsets.into_iter().enumerate() {
            if group1_start < 0 {
                continue;
            }

            let group1_end =
                self.game_state
                    .particle_offsets
                    .into_iter()
                    .skip(i + 1)
                    .find(|&item| item > 0)
                    .unwrap_or(self.game_state.particle_data.len() as i32) as usize;

            for (j, group2_start) in self.game_state.particle_offsets.into_iter().enumerate() {
                if group2_start < 0 {
                    continue;
                }

                let group2_end = self
                    .game_state
                    .particle_offsets
                    .into_iter()
                    .skip(j + 1)
                    .find(|&item| item > 0)
                    .unwrap_or(self.game_state.particle_data.len() as i32)
                    as usize;

                let new_particle_data = interaction(
                    &self.game_state.particle_data[group1_start as usize..group1_end],
                    &self.game_state.particle_data[group2_start as usize..group2_end],
                    self.game_state.power_slider.col(i)[j],
                    self.game_state.r_slider.col(i)[j],
                    0.1f32,
                );
                self.game_state
                    .particle_data
                    .splice(group1_start as usize..group1_end, new_particle_data);
            }
        }

        self.queue.as_ref().unwrap().write_buffer(
            self.particle_buffer.as_mut().unwrap(),
            0,
            bytemuck::cast_slice(&self.game_state.particle_data),
        );
    }

    pub fn main_loop<T>(
        &mut self,
        event: Event<()>,
        _target: &EventLoopWindowTarget<T>,
        control_flow: &mut ControlFlow,
        last_update_inst: &mut Instant,
    ) {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let render_pipeline = self.render_pipeline.as_ref().unwrap();
        let particle_buffer = self.particle_buffer.as_ref().unwrap();
        let vertices_buffer = self.vertex_buffer.as_ref().unwrap();

        let num_particles = self.game_state.particle_data.len();

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                self.config.as_mut().unwrap().width = size.width;
                self.config.as_mut().unwrap().height = size.height;
                self.surface
                    .configure(&device, &self.config.as_ref().unwrap());
                // On macos the window needs to be redrawn manually after resizing
                // self.window.request_redraw();
            }
            Event::RedrawEventsCleared => {
                let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE as f64);
                let time_since_last_frame = last_update_inst.elapsed();

                if time_since_last_frame >= target_frametime {
                    *last_update_inst = Instant::now();
                    log::info!(
                        "Framerate: {}",
                        1000u128 / time_since_last_frame.as_millis()
                    );

                    self.update();
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
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
                    rpass.set_pipeline(render_pipeline);
                    rpass.set_index_buffer(
                        self.index_buffer.as_ref().unwrap().slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    rpass.set_vertex_buffer(0, particle_buffer.slice(..));
                    rpass.set_vertex_buffer(1, vertices_buffer.slice(..));
                    // rpass.draw(0..6, 0..NUM_PARTICLES);
                    rpass.draw_indexed(0..6 as u32, 0, 0..num_particles as u32);
                    // device.as_ref().unwrap().poll(wgpu::Maintain::Wait);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        };
    }
}
