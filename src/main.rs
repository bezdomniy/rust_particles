mod app;

use app::App;
use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    env_logger::init();

    let mut particles_app = App::new(&window);

    pollster::block_on(particles_app.setup());
    pollster::block_on(particles_app.run(event_loop));
}
