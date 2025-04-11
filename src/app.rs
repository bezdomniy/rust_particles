use egui::Color32;
use glam::{Mat4, UVec4, Vec2};
use rand::{distr::Uniform, rng, Rng};
use std::{borrow::Cow, f32::EPSILON};

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
    wgpu::{util::DeviceExt, BindGroup},
};

use crate::bvh::Bvh;

const FRAMERATE: u32 = 30;
const BOUNDS_TOGGLE: bool = true;
const PARTICLE_SIZE: f32 = 2f32;
// const PARTICLES_PER_GROUP: u32 = 64;
// const MAX_VELOCITY: f32 = 1f32;
// const MAX_VELOCITY: f32 = 0.1f32;
const INITIAL_VISCOSITY: f32 = 0.9990234375;
const REPULSE_RADIUS: f32 = 0.25;
const USE_LINEAR_BVH: bool = false;

pub struct App {
    game_state: GameState,
    last_update_inst: Instant,
    _target_frame_time: Duration,
}

#[derive(Clone)]
struct GameState {
    particle_data: Vec<Particle>,
    particle_offsets: [i32; 4],
    pub power_slider: Mat4,
    pub r_slider: Mat4,
    pub num_particles: UVec4,
    pub viscosity: f32,
}

impl GameState {
    fn init_particles(&mut self) {
        let spread = 0.8f32;
        let aspect_ratio = 4f32 / 3f32;
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
                let random_pos = match i {
                    // 0 => Vec2::new(
                    //     rng().random_range(-1f32..=0f32),
                    //     rng().random_range(0f32..=1f32),
                    // ),
                    // 1 => Vec2::new(
                    //     rng().random_range(0f32..=1f32),
                    //     rng().random_range(0f32..=1f32),
                    // ),
                    // 2 => Vec2::new(
                    //     rng().random_range(0f32..=1f32),
                    //     rng().random_range(-1f32..=0f32),
                    // ),
                    // 3 => Vec2::new(
                    //     rng().random_range(-1f32..=0f32),
                    //     rng().random_range(-1f32..=0f32),
                    // ),
                    _ => Vec2::new(
                        rng().random_range(-spread * aspect_ratio..=spread * aspect_ratio),
                        rng().random_range(-spread..=spread),
                    ),
                };
                let particle = Particle {
                    // pos: Vec2::new(
                    //     rng().random_range(-1f32..=1f32),
                    //     rng().random_range(-1f32..=1f32),
                    // ),
                    pos: random_pos,
                    // vel: Vec2::new(
                    //     rng().random_range(-0.01f32..=0.01f32),
                    //     rng().random_range(-0.01f32..=0.01f32),
                    // ),
                    vel: Vec2::new(0f32, 0f32),
                    cls: i as u32,
                };
                self.particle_data.push(particle);
            }
        }
    }
    fn update(&mut self, aspect_ratio: f32) {
        for (i, group1_start) in self.particle_offsets.into_iter().enumerate() {
            if group1_start < 0 {
                continue;
            }

            let group1_end = self
                .particle_offsets
                .into_iter()
                .skip(i + 1)
                .find(|&item| item > 0)
                .unwrap_or(self.particle_data.len() as i32) as usize;

            for (j, group2_start) in self.particle_offsets.into_iter().enumerate() {
                if group2_start < 0 {
                    continue;
                }

                let group2_end =
                    self.particle_offsets
                        .into_iter()
                        .skip(j + 1)
                        .find(|&item| item > 0)
                        .unwrap_or(self.particle_data.len() as i32) as usize;

                let bvh = Bvh::new(
                    &mut self.particle_data[group2_start as usize..group2_end],
                    self.r_slider.col(i)[j],
                    USE_LINEAR_BVH,
                );

                interaction(
                    &mut self.particle_data,
                    &bvh,
                    group1_start as usize,
                    group1_end,
                    group2_start as usize,
                    group2_end,
                    self.power_slider.col(i)[j],
                    self.r_slider.col(i)[j],
                    self.viscosity,
                    aspect_ratio,
                );
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub pos: Vec2,
    vel: Vec2,
    cls: u32,
}

// Interaction between 2 particle groups
fn interaction(
    particles: &mut Vec<Particle>,
    bvh: &Bvh,
    group1_start: usize,
    group1_end: usize,
    group2_start: usize,
    group2_end: usize,
    g: f32,
    radius: f32,
    viscosity: f32,
    aspect_ratio: f32,
) {
    let group2 = &particles.clone()[group2_start as usize..group2_end];
    let group1 = &mut particles[group1_start as usize..group1_end];
    #[cfg(target_arch = "wasm32")]
    let g_iter = group1.iter_mut();
    #[cfg(not(target_arch = "wasm32"))]
    let g_iter = group1.par_iter_mut();

    g_iter.for_each(|p1| {
        let f = bvh.intersect(p1, radius, g, REPULSE_RADIUS, group2);

        p1.vel += f;
        p1.vel *= 1f32 - viscosity;

        // p1.vel = p1.vel.clamp_length_max(MAX_VELOCITY);

        if BOUNDS_TOGGLE {
            //not good enough! Need fixing
            if (p1.pos.x >= aspect_ratio) || (p1.pos.x <= -aspect_ratio) {
                p1.vel.x *= -1f32;
                p1.pos.x = (aspect_ratio - EPSILON) * p1.pos.x.signum();
            }
            if (p1.pos.y >= 1f32) || (p1.pos.y <= -1f32) {
                p1.vel.y *= -1f32;
                p1.pos.y = 1f32 * p1.pos.y.signum();
            }
        }

        p1.pos += p1.vel;
    })
}

impl App {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Self {
        let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
        let device = &wgpu_render_state.device;

        let mut rng = rand::rng();

        // let power_vals: [f32; 16] = (0..16)
        //     .map(|_| rng.sample(Uniform::new(-0.5f32, 0.5f32)))
        //     .collect::<Vec<f32>>()
        //     .try_into()
        //     .unwrap();

        let power_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(Uniform::new(-1f32, 1f32).unwrap()))
            // .map(|_| 0f32)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let r_vals: [f32; 16] = (0..16)
            .map(|_| rng.sample(Uniform::new(0.01f32, 0.3f32).unwrap()))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        #[cfg(target_arch = "wasm32")]
        let num_particles = UVec4::new(3000, 3000, 3000, 3000);

        #[cfg(not(target_arch = "wasm32"))]
        // let num_particles = UVec4::new(5000, 5000, 5000, 5000);
        // let num_particles = UVec4::new(50, 50, 50, 50);
        let num_particles = UVec4::new(10000, 10000, 10000, 10000);

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
            viscosity: INITIAL_VISCOSITY,
        };

        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/particles.wgsl"))),
        });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let transform = Mat4::IDENTITY;
        let mx_ref: &[f32; 16] = transform.as_ref();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: None,
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
                entry_point: Some("main_vs"),
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
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("main_fs"),
                targets: &[Some(swapchain_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: Default::default(),
        });

        wgpu_render_state
            .renderer
            .write()
            // .paint_callback_resources
            .callback_resources
            .insert(RenderResources {
                render_bind_group: bind_group,
                render_pipeline,
                index_buffer,
                uniform_buffer,
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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Window::new("Parameters")
                .default_open(false)
                .show(ctx, |ui| {
                    ui.horizontal_top(|ui| {
                        egui::Grid::new("power_slider").show(ui, |ui| {
                            ui.label("Power");
                            ui.label("Red");
                            ui.label("Green");
                            ui.label("Blue");
                            ui.label("White");
                            ui.end_row();

                            ui.label("Red");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.x_axis.x)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.x_axis.y)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.x_axis.z)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.x_axis.w)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("Green");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.y_axis.x)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.y_axis.y)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.y_axis.z)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.y_axis.w)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("Blue");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.z_axis.x)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.z_axis.y)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.z_axis.z)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.z_axis.w)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("White");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.w_axis.x)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.w_axis.y)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.w_axis.z)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.power_slider.w_axis.w)
                                    .range(-1f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();
                        });
                        egui::Grid::new("r_slider").show(ui, |ui| {
                            ui.label("Radius");
                            ui.label("Red");
                            ui.label("Green");
                            ui.label("Blue");
                            ui.label("White");
                            ui.end_row();
                            ui.label("Red");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.x_axis.x)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.x_axis.y)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.x_axis.z)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.x_axis.w)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("Green");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.y_axis.x)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.y_axis.y)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.y_axis.z)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.y_axis.w)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();
                            ui.label("Blue");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.z_axis.x)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.z_axis.y)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.z_axis.z)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.z_axis.w)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("White");
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.w_axis.x)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.w_axis.y)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.w_axis.z)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.r_slider.w_axis.w)
                                    .range(0f32..=1f32)
                                    .speed(0.01),
                            );
                            ui.end_row();
                        });
                        egui::Grid::new("power_slider").show(ui, |ui| {
                            ui.label("Viscosity");
                            ui.end_row();
                            ui.add(
                                egui::DragValue::new(&mut self.game_state.viscosity)
                                    .range(0f32..=1f32)
                                    .speed(0.00001),
                            );
                            ui.end_row();
                        });
                    })
                });

            egui::Frame::canvas(ui.style())
                .fill(Color32::BLACK)
                .show(ui, |ui| {
                    self.game_state
                        .update(ui.available_width() / ui.available_height());

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
    // particle_data: Vec<Particle>,
    game_state: GameState,
}

impl egui_wgpu::CallbackTrait for CustomCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &RenderResources = resources.get().unwrap();
        resources.prepare(
            device,
            queue,
            &self.game_state,
            screen_descriptor.size_in_pixels,
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &RenderResources = resources.get().unwrap();
        resources.paint(render_pass, self.game_state.particle_data.len());
    }
}

impl App {
    fn draw_app(&mut self, ui: &mut egui::Ui) {
        let (rect, _response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            CustomCallback {
                game_state: self.game_state.clone(),
            },
        ));
        self.last_update_inst = Instant::now();
    }
}

struct RenderResources {
    render_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    uniform_buffer: Buffer,
    particle_buffer: Buffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

impl RenderResources {
    fn prepare(&self, _device: &Device, queue: &Queue, game_state: &GameState, size: [u32; 2]) {
        let aspect_ratio = size[0] as f32 / size[1] as f32;

        let transform =
            Mat4::orthographic_rh(-aspect_ratio, aspect_ratio, -1f32, 1f32, -1f32, 1f32);

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&transform.to_cols_array()),
        );

        queue.write_buffer(
            &self.particle_buffer,
            0,
            bytemuck::cast_slice(game_state.particle_data.as_slice()),
        );
    }

    fn paint(&self, render_pass: &mut RenderPass<'_>, num_particles: usize) {
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..6u32, 0, 0..num_particles as u32);
    }
}
