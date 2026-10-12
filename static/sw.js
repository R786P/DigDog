// PillaArt service worker — minimal, just enough for installability.
const CACHE_NAME = "pillaart-cache-v1";
const CORE_ASSETS = [
  "/",
  "/static/style.css",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Network-first for pages (so AI-generated content stays fresh), cache fallback for static assets.
self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  if (req.url.includes("/static/")) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req))
    );
  }
});
