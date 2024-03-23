use egui::Color32;
use glam::{Mat4, UVec4, Vec2};
use rand::{distributions::Uniform, thread_rng, Rng};
use std::borrow::Cow;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
// #[cfg(target_arch = "wasm32")]
// pub use wasm_bindgen_rayon::init_thread_pool;

use eframe::{
    egui_wgpu::{
        self,
        wgpu::{self, Buffer, Device, Queue, RenderPass, RenderPipeline},
        ScreenDescriptor,
    },
    wgpu::util::DeviceExt,
};

const FRAMERATE: u32 = 30;
const BOUNDS_TOGGLE: bool = true;
const PARTICLE_SIZE: f32 = 2f32;
// const PARTICLES_PER_GROUP: u32 = 64;
const MAX_VELOCITY: f32 = 0.1f32;

pub struct App {
    game_state: GameState,
    last_update_inst: Instant,
    _target_frame_time: Duration,
}

struct GameState {
    particle_data: Vec<Particle>,
    particle_offsets: [i32; 4],
    pub power_slider: Mat4,
    pub r_slider: Mat4,
    pub num_particles: UVec4,
}

impl GameState {
    fn init_particles(&mut self) {
        for i in 0..4 {
            if self.particle_offsets[i] < 0 {
                continue;
            }

            let start = self.particle_offsets[i];
            let end = self
                .particle_offsets
                .into_iter()
                .skip(i + 1)
                .find(|&item| item > 0)
                .unwrap_or(
                    (self.num_particles.x
                        + self.num_particles.y
                        + self.num_particles.z
                        + self.num_particles.w) as i32,
                );

            for _ in start..end {
                self.particle_data.push(Particle {
                    pos: Vec2::new(
                        thread_rng().gen_range(-1f32..=1f32),
                        thread_rng().gen_range(-1f32..=1f32),
                    ),
                    vel: Vec2::new(
                        thread_rng().gen_range(-0.001f32..=0.001f32),
                        thread_rng().gen_range(-0.001f32..=0.001f32),
                    ),
                    cls: i as u32,
                });
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
    cls: u32,
}

// Interaction between 2 particle groups
fn interaction(group1: &mut [Particle], group2: &[Particle], g: f32, radius: f32, viscosity: f32) {
    #[cfg(target_arch = "wasm32")]
    let g_iter = group1.iter();
    #[cfg(not(target_arch = "wasm32"))]
    let g_iter = group1.par_iter_mut();

    g_iter.for_each(|p1| {
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
                f += (g * d.normalize()) / r;
            }
        });

        if f.length() >= f32::EPSILON {
            // let mut vel = p1.vel;
            // let mut vel = (p1.vel + (f * g)) * (1f32 - viscosity);
            let mut vel = (p1.vel * (1f32 - viscosity)) + (f * 0.00001f32);

            if vel.length() > MAX_VELOCITY {
                vel = vel.normalize() * MAX_VELOCITY;
            }

            if BOUNDS_TOGGLE {
                //not good enough! Need fixing
                if (p1.pos.x >= 1f32) || (p1.pos.x <= -1f32) {
                    vel.x *= -1f32;
                }
                if (p1.pos.y >= 1f32) || (p1.pos.y <= -1f32) {
                    vel.y *= -1f32;
                }
            }
            p1.pos += vel;
            p1.vel = vel;
        }
    })
}

impl App {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Self {
        let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
        let device = &wgpu_render_state.device;

        let mut rng = rand::thread_rng();

        let power_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(Uniform::new(-1f32, 1f32)))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let r_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(Uniform::new(0.01f32, 0.9f32)))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        #[cfg(target_arch = "wasm32")]
        let num_particles = UVec4::new(1000, 1000, 1000, 1000);

        #[cfg(not(target_arch = "wasm32"))]
        let num_particles = UVec4::new(3000, 3000, 3000, 3000);

        let mut game_state = GameState {
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

        // let swapchain_format = self.surface.get_supported_formats(&adapter)[0];
        let swapchain_format = wgpu_render_state.target_format;

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

        game_state.init_particles();

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&game_state.particle_data),
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

        wgpu_render_state
            .renderer
            .write()
            // .paint_callback_resources
            .callback_resources
            .insert(RenderResources {
                render_pipeline,
                index_buffer,
                particle_buffer,
                vertex_buffer,
            });

        Self {
            game_state,
            last_update_inst: Instant::now(),
            _target_frame_time: Duration::from_secs_f64(1.0 / FRAMERATE as f64),
        }

        // self.config = Some(wgpu::SurfaceConfiguration {
        //     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        //     format: swapchain_format,
        //     width: physical_size.width,
        //     height: physical_size.height,
        //     present_mode: wgpu::PresentMode::Fifo,
        // });

        // self.surface
        //     .configure(&device, &self.config.as_ref().unwrap());

        // self.device = Some(device);
        // self.queue = Some(queue);
        // self.adapter = Some(adapter);
        // self.render_pipeline = Some(render_pipeline);
        // self.particle_buffer = Some(particle_buffer);
        // self.vertex_buffer = Some(vertex_buffer);
        // self.index_buffer = Some(index_buffer);
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

                let particles_data_copy = self.game_state.particle_data.clone();

                interaction(
                    &mut self.game_state.particle_data[group1_start as usize..group1_end],
                    &particles_data_copy[group2_start as usize..group2_end],
                    self.game_state.power_slider.col(i)[j],
                    self.game_state.r_slider.col(i)[j],
                    0.5f32,
                );
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::canvas(ui.style())
                .fill(Color32::BLACK)
                .show(ui, |ui| {
                    self.update();
                    log::info!(
                        "FPS: {:?}",
                        1000u128 / self.last_update_inst.elapsed().as_millis()
                    );
                    self.draw_app(ui);
                    ctx.request_repaint();
                })
        });
    }
}

struct CustomCallback {
    particle_data: Vec<Particle>,
}

impl egui_wgpu::CallbackTrait for CustomCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &RenderResources = resources.get().unwrap();
        resources.prepare(device, queue, &self.particle_data);
        Vec::new()
    }

    fn paint<'a>(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        resources: &'a egui_wgpu::CallbackResources,
    ) {
        let resources: &RenderResources = resources.get().unwrap();
        resources.paint(render_pass, self.particle_data.len());
    }
}

impl App {
    fn draw_app(&mut self, ui: &mut egui::Ui) {
        let (rect, _response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            CustomCallback {
                particle_data: self.game_state.particle_data.clone(),
            },
        ));
        self.last_update_inst = Instant::now();
    }
}

struct RenderResources {
    render_pipeline: RenderPipeline,
    particle_buffer: Buffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

impl RenderResources {
    fn prepare(&self, _device: &Device, queue: &Queue, particle_data: &[Particle]) {
        queue.write_buffer(
            &self.particle_buffer,
            0,
            bytemuck::cast_slice(particle_data),
        );
    }

    fn paint<'rp>(&'rp self, render_pass: &mut RenderPass<'rp>, num_particles: usize) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..6u32, 0, 0..num_particles as u32);
    }
}
