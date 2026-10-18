// Bump the version to drop caches written by older builds.
var cacheName = 'rust-particles-v2';
var filesToCache = [
    './',
    './index.html',
    './rust_particles.js',
    './rust_particles_bg.wasm',
];

/* Start the service worker and cache all of the app's content */
self.addEventListener('install', function (e) {
    self.skipWaiting();
    e.waitUntil(
        caches.open(cacheName).then(function (cache) {
            return cache.addAll(filesToCache);
        })
    );
});

/* Remove caches from previous versions (including the old egui template name) */
self.addEventListener('activate', function (e) {
    e.waitUntil(
        caches.keys().then(function (keys) {
            return Promise.all(keys.filter(function (key) {
                return key !== cacheName && (key.startsWith('rust-particles-') || key === 'egui-template-pwa');
            }).map(function (key) {
                return caches.delete(key);
            }));
        }).then(function () {
            return self.clients.claim();
        })
    );
});

/* Network first so new builds show up immediately; fall back to the cache when offline */
self.addEventListener('fetch', function (e) {
    if (e.request.method !== 'GET') return;
    e.respondWith(
        fetch(e.request).then(function (response) {
            if (response.ok) {
                var copy = response.clone();
                caches.open(cacheName).then(function (cache) {
                    cache.put(e.request, copy);
                });
            }
            return response;
        }).catch(function () {
            return caches.match(e.request);
        })
    );
});
