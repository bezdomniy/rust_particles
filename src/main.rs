use std::time::Instant;
use winit::event_loop::EventLoop;
mod app;

use app::App;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    let mut particles_app = App::new(&window);
    // particles_app.setup(window.inner_size()).await;
    pollster::block_on(particles_app.setup(window.inner_size()));

    let mut now = Instant::now();

    event_loop.run(move |event, target, control_flow| {
        particles_app.main_loop(event, target, control_flow, &mut now)
    });
}
