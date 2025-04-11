#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let options = eframe::NativeOptions::default();

    let _ = eframe::run_native(
        "Particles",
        options,
        Box::new(|cc| Ok(Box::new(rust_particles::App::new(cc)))),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;
    
    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let options = eframe::WebOptions::default();

    let document = web_sys::window()
        .expect("No window")
        .document()
        .expect("No document");

    let canvas = document
        .get_element_by_id("canvas")
        .expect("Failed to find canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("canvas was not a HtmlCanvasElement");

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                canvas,
                options,
                Box::new(|cc| Ok(Box::new(rust_particles::App::new(cc)))),
            )
            .await
            .expect("failed to start eframe");
    });
}
