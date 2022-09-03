mod app;

use app::App;

fn main() {
    env_logger::init();

    let mut particles_app = App::new();

    pollster::block_on(particles_app.setup());
    pollster::block_on(particles_app.run());
}
