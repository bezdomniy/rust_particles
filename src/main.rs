use winit::event_loop::EventLoop;

mod app;

use app::App;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let mut particles_app = App::new(&event_loop);

    pollster::block_on(particles_app.setup());
    pollster::block_on(particles_app.run(event_loop));
}
