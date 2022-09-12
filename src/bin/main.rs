use rust_particles::App;
use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new();
    let mut particles_app = App::new(&event_loop);

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();

        pollster::block_on(particles_app.setup());
        pollster::block_on(particles_app.run(event_loop));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");

        wasm_bindgen_futures::spawn_local(particles_app.setup());
        wasm_bindgen_futures::spawn_local(particles_app.run(event_loop));
    }
}
