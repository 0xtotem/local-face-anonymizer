import * as ort from 'onnxruntime-web/webgpu';

type Face = { x1: number; y1: number; x2: number; y2: number; score: number };
type WorkerRequest = { type: 'init'; modelUrl: string; wasmBase: string } | { type: 'detect'; frame: ImageBitmap; id: number };

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: unknown) => void;
};
const detectionThreshold = 0.1;
let sessionPromise: Promise<ort.InferenceSession> | undefined;
let provider = 'WASM';
let modelUrl = '';

function initSession() {
  sessionPromise ??= (async () => {
    let session: ort.InferenceSession | undefined;
    if ('gpu' in navigator) {
      try {
        session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'] });
        provider = 'WebGPU';
      } catch {
        session = undefined;
      }
    }
    if (!session) {
      provider = 'WASM';
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    }
    return session;
  })();
  return sessionPromise;
}

function planeIndex(channel: number, rows: number, columns: number, row: number, column: number) {
  return channel * rows * columns + row * columns + column;
}

function iou(a: Face, b: Face) {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = width * height;
  return intersection / Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection);
}

async function detect(frame: ImageBitmap) {
  const originalWidth = frame.width;
  const originalHeight = frame.height;
  const resize = Math.min(1, 768 / Math.max(originalWidth, originalHeight));
  const width = Math.max(32, Math.floor(originalWidth * resize / 32) * 32);
  const height = Math.max(32, Math.floor(originalHeight * resize / 32) * 32);
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Could not create the video detection canvas.');
  context.drawImage(frame, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const plane = width * height;
  const tensorData = new Float32Array(3 * plane);
  for (let pixel = 0; pixel < plane; pixel++) {
    tensorData[pixel] = rgba[pixel * 4];
    tensorData[plane + pixel] = rgba[pixel * 4 + 1];
    tensorData[plane * 2 + pixel] = rgba[pixel * 4 + 2];
  }
  const session = await initSession();
  const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', tensorData, [1, 3, height, width]) });
  const heat = outputs['537'];
  const scale = outputs['538'];
  const offset = outputs['539'];
  if (!heat || !scale || !offset) throw new Error('The detector returned an unexpected output.');
  const rows = heat.dims.at(-2)!;
  const columns = heat.dims.at(-1)!;
  const heatData = heat.data as Float32Array;
  const scaleData = scale.data as Float32Array;
  const offsetData = offset.data as Float32Array;
  const candidates: Face[] = [];
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
    const score = heatData[row * columns + column];
    if (score <= detectionThreshold) continue;
    const boxWidth = Math.exp(scaleData[planeIndex(1, rows, columns, row, column)]) * 4;
    const boxHeight = Math.exp(scaleData[planeIndex(0, rows, columns, row, column)]) * 4;
    const centerX = (column + offsetData[planeIndex(1, rows, columns, row, column)] + 0.5) * 4;
    const centerY = (row + offsetData[planeIndex(0, rows, columns, row, column)] + 0.5) * 4;
    candidates.push({
      x1: Math.max(0, centerX - boxWidth / 2) / width * originalWidth,
      y1: Math.max(0, centerY - boxHeight / 2) / height * originalHeight,
      x2: Math.min(width, centerX + boxWidth / 2) / width * originalWidth,
      y2: Math.min(height, centerY + boxHeight / 2) / height * originalHeight,
      score,
    });
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept: Face[] = [];
  for (const candidate of candidates) if (kept.every((face) => iou(face, candidate) < 0.3)) kept.push(candidate);
  return kept;
}

workerScope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === 'init') {
    modelUrl = request.modelUrl;
    ort.env.wasm.wasmPaths = request.wasmBase;
    try {
      await initSession();
      workerScope.postMessage({ type: 'ready', provider });
    } catch (error) {
      sessionPromise = undefined;
      workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  try {
    const boxes = await detect(request.frame);
    workerScope.postMessage({ type: 'result', id: request.id, boxes, provider });
  } catch (error) {
    workerScope.postMessage({ type: 'error', id: request.id, message: error instanceof Error ? error.message : String(error) });
  } finally {
    request.frame.close();
  }
};
