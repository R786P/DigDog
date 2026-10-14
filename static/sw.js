// PillaArt service worker — minimal, just enough for installability.
// IMPORTANT: bump CACHE_NAME every time you change static files (CSS/JS/icons)
// so old browsers/phones drop the stale cache and pick up the new version.
const CACHE_NAME = "pillaart-cache-v7";
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

// Network-first everywhere: always try to fetch the latest file first.
// Only fall back to the cached copy if the network request fails (e.g. offline).
// This guarantees that after every redeploy, the newest CSS/HTML/JS loads immediately.
self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  if (!req.url.includes("/static/")) return; // only manage static assets

  event.respondWith(
    fetch(req)
      .then((res) => {
        const resClone = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, resClone));
        return res;
      })
      .catch(() => caches.match(req))
  );
});
