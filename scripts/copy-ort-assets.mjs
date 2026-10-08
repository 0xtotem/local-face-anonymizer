import { cp, mkdir, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(root, 'node_modules/onnxruntime-web/dist');
const destination = join(root, 'public/ort');
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
for (const asset of [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
]) {
  await cp(join(source, asset), join(destination, asset));
}
