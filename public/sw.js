const CACHE = 'covered-faces-v7';

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(['./', './manifest.webmanifest']);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys
      .filter((key) => key.startsWith('covered-faces-') && key !== CACHE)
      .map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cached = await caches.match(request);

    // Always check the deployed document first so it points at the current hashed bundles.
    if (request.mode === 'navigate') {
      try {
        const response = await fetch(request, { cache: 'no-cache' });
        if (response.ok && response.type === 'basic') {
          const cache = await caches.open(CACHE);
          await cache.put(request, response.clone()).catch(() => undefined);
        }
        return response;
      } catch {
        return cached || await caches.match('./') || Response.error();
      }
    }

    if (cached) return cached;

    try {
      const response = await fetch(request);
      if (response.ok && response.type === 'basic') {
        const cache = await caches.open(CACHE);
        await cache.put(request, response.clone()).catch(() => undefined);
      }
      return response;
    } catch {
      return new Response('This resource is not available offline.', {
        status: 503,
        statusText: 'Service Unavailable',
      });
    }
  })());
});
