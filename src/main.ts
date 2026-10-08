import * as ort from 'onnxruntime-web/webgpu';
import * as exifr from 'exifr';
import './style.css';

type Face = { x1: number; y1: number; x2: number; y2: number; score: number; blur?: boolean };
type DetectionResult = { boxes: Face[]; provider: string };
const detectionThreshold = 0.2;

const input = document.querySelector<HTMLInputElement>('#file-input')!;
const dropzone = document.querySelector<HTMLLabelElement>('#dropzone')!;
const detectButton = document.querySelector<HTMLButtonElement>('#detect-button')!;
const exportButton = document.querySelector<HTMLButtonElement>('#export-button')!;
const status = document.querySelector<HTMLParagraphElement>('#status')!;
const engineStatus = document.querySelector<HTMLSpanElement>('#engine-status')!;
const canvas = document.querySelector<HTMLCanvasElement>('#preview-canvas')!;
const stage = document.querySelector<HTMLDivElement>('#preview-stage')!;
const comparisonControl = document.querySelector<HTMLDivElement>('#comparison-control')!;
const comparisonDivider = document.querySelector<HTMLDivElement>('#comparison-divider')!;
const comparisonRange = document.querySelector<HTMLInputElement>('#comparison-range')!;
const faceMarkers = document.querySelector<HTMLDivElement>('#face-markers')!;
const revealedFaces = document.querySelector<HTMLUListElement>('#revealed-faces')!;
const editEmpty = document.querySelector<HTMLSpanElement>('#edit-empty')!;
const emptyPreview = document.querySelector<HTMLDivElement>('#empty-preview')!;
const faceCount = document.querySelector<HTMLDivElement>('#face-count')!;
const imageSize = document.querySelector<HTMLSpanElement>('#image-size')!;
const videoPreview = document.querySelector<HTMLVideoElement>('#video-preview')!;
const processedVideo = document.querySelector<HTMLVideoElement>('#processed-video')!;
const previewCaption = document.querySelector<HTMLSpanElement>('#preview-caption')!;
const metadataPanel = document.querySelector<HTMLElement>('#metadata-panel')!;
const metadataBefore = document.querySelector<HTMLDListElement>('#metadata-before')!;
const metadataAfter = document.querySelector<HTMLDListElement>('#metadata-after')!;
const metadataStatus = document.querySelector<HTMLSpanElement>('#metadata-status')!;
const context = canvas.getContext('2d', { willReadFrequently: true })!;
const canvasBlurSupported = (() => {
  const probe = document.createElement('canvas');
  probe.width = 5;
  probe.height = 5;
  const probeContext = probe.getContext('2d');
  if (!probeContext) return false;
  const source = document.createElement('canvas');
  source.width = 1;
  source.height = 1;
  const sourceContext = source.getContext('2d');
  if (!sourceContext) return false;
  try {
    sourceContext.fillStyle = '#fff';
    sourceContext.fillRect(0, 0, 1, 1);
    probeContext.filter = 'blur(2px)';
    probeContext.drawImage(source, 2, 2);
    return probeContext.getImageData(1, 2, 1, 1).data[3] > 0;
  } catch {
    return false;
  }
})();

let image: HTMLImageElement | undefined;
let sourceFile: File | undefined;
let videoUrl: string | undefined;
let processedVideoUrl: string | undefined;
let isVideo = false;
let videoRecorder: MediaRecorder | undefined;
let videoChunks: Blob[] = [];
let videoFrame = 0;
let videoBusy = false;
let videoDetector: Worker | undefined;
let videoDetectInterval = 0;
let videoRequestId = 0;
let videoFailureMessage = '';
let videoBlurCanvas = document.createElement('canvas');
let videoBlurContext = videoBlurCanvas.getContext('2d');
let videoAudioContext: AudioContext | undefined;
let videoAudioSource: MediaElementAudioSourceNode | undefined;
let videoAudioDestination: MediaStreamAudioDestinationNode | undefined;
let filename = 'photo';
let detections: Face[] = [];
let processed: HTMLCanvasElement | undefined;
const provider = 'WASM';
let sessionPromise: Promise<ort.InferenceSession> | undefined;
let comparisonPercent = 50;
let comparisonFrame = 0;
let metadataGeneration = 0;

const metadataLabels: Record<string, string> = {
  Make: 'Camera maker', Model: 'Camera model', LensModel: 'Lens',
  DateTimeOriginal: 'Date taken', CreateDate: 'Digitized', ModifyDate: 'Modified',
  GPSLatitude: 'GPS latitude', GPSLongitude: 'GPS longitude', GPSAltitude: 'GPS altitude',
  GPSDateStamp: 'GPS date', GPSProcessingMethod: 'GPS method',
  Software: 'Software', ImageDescription: 'Description', Artist: 'Artist', Copyright: 'Copyright',
  Orientation: 'Orientation', FNumber: 'Aperture', ExposureTime: 'Exposure time',
  ISO: 'ISO', FocalLength: 'Focal length', BodySerialNumber: 'Camera serial number',
  SerialNumber: 'Serial number', LensSerialNumber: 'Lens serial number', ImageUniqueID: 'Image ID',
};

function formatMetadataValue(value: unknown): string {
  if (value instanceof Date) return value.toLocaleString();
  if (Array.isArray(value)) return value.map(formatMetadataValue).join(', ');
  if (typeof value === 'number') return Number.isFinite(value) ? String(Number(value.toFixed(6))) : '—';
  if (value && typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function addMetadataRow(list: HTMLDListElement, label: string, value: string) {
  const term = document.createElement('dt');
  term.textContent = label;
  const description = document.createElement('dd');
  description.textContent = value;
  list.append(term, description);
}

function showMetadata(list: HTMLDListElement, file: Pick<File, 'name' | 'type' | 'size'>, width: number | undefined, height: number | undefined, tags: Record<string, unknown> | undefined, error?: string) {
  list.replaceChildren();
  addMetadataRow(list, 'File', file.name);
  addMetadataRow(list, 'Format', file.type || file.name.split('.').pop()?.toUpperCase() || 'Unknown');
  addMetadataRow(list, 'File size', `${(file.size / 1024).toFixed(1)} KB`);
  if (width && height) addMetadataRow(list, 'Dimensions', `${width} × ${height} px`);
  const found = Object.entries(metadataLabels).filter(([key]) => tags?.[key] !== undefined && tags[key] !== null);
  for (const [key, label] of found) addMetadataRow(list, label, formatMetadataValue(tags?.[key]));
  if (error || found.length === 0) {
    const note = document.createElement('div');
    note.className = 'metadata-empty';
    note.textContent = error || 'No selected EXIF, GPS, or camera fields found.';
    list.append(note);
  }
}

async function readMetadata(file: Blob) {
  try {
    return await exifr.parse(file, true) as Record<string, unknown> | undefined;
  } catch (error) {
    console.info('Metadata could not be fully parsed.', error);
    return undefined;
  }
}

function syncComparisonBounds() {
  if (comparisonControl.hidden || canvas.hidden) return;
  const stageBounds = stage.getBoundingClientRect();
  const canvasBounds = canvas.getBoundingClientRect();
  comparisonControl.style.left = `${canvasBounds.left - stageBounds.left}px`;
  comparisonControl.style.top = `${canvasBounds.top - stageBounds.top}px`;
  comparisonControl.style.width = `${canvasBounds.width}px`;
  comparisonControl.style.height = `${canvasBounds.height}px`;
  faceMarkers.style.clipPath = `inset(0 0 0 ${comparisonPercent}%)`;
}

const comparisonResizeObserver = new ResizeObserver(syncComparisonBounds);
comparisonResizeObserver.observe(stage);

function setStatus(message: string, busy = false) {
  status.textContent = message;
  status.classList.toggle('busy', busy);
}

async function loadImage(file: File) {
  const generation = ++metadataGeneration;
  const isHeicFile = /heic|heif/i.test(file.type) || /\.(heic|heif)$/i.test(file.name);
  if (!file.type.startsWith('image/') && !isHeicFile) {
    setStatus('Unsupported file. Choose a photo. Video processing is temporarily disabled.');
    return;
  }
  metadataPanel.hidden = false;
  metadataStatus.textContent = 'Reading metadata locally…';
  metadataBefore.replaceChildren();
  metadataAfter.replaceChildren();
  const placeholder = document.createElement('div');
  placeholder.className = 'metadata-empty';
  placeholder.textContent = 'Run anonymization to inspect the exported PNG.';
  metadataAfter.append(placeholder);
  const metadataPromise = readMetadata(file);
  sourceFile = file;
  if (isHeicFile) setStatus('Converting HEIC locally…', true);
  let previewBlob: Blob = file;
  if (isHeicFile) {
    try {
      const { heicTo } = await import('heic-to');
      previewBlob = await heicTo({ blob: file, type: 'image/png' });
    } catch (error) {
      setStatus(`This HEIC image could not be decoded: ${error instanceof Error ? error.message : String(error)}`);
      metadataStatus.textContent = 'HEIC decoding failed';
      const tags = await metadataPromise;
      showMetadata(metadataBefore, file, undefined, undefined, tags);
      return;
    }
  }
  const objectUrl = URL.createObjectURL(previewBlob);
  const nextImage = new Image();
  nextImage.onload = () => {
    if (generation !== metadataGeneration) {
      URL.revokeObjectURL(objectUrl);
      return;
    }
    if (image?.src.startsWith('blob:')) URL.revokeObjectURL(image.src);
    image = nextImage;
    filename = file.name.replace(/\.[^.]+$/, '') || 'photo';
    detections = [];
    processed = undefined;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.hidden = false;
    emptyPreview.hidden = true;
    stage.classList.add('has-image');
    imageSize.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    showMetadata(metadataBefore, file, image.naturalWidth, image.naturalHeight, undefined, 'Reading embedded metadata…');
    void metadataPromise.then((tags) => {
      if (generation === metadataGeneration) {
        showMetadata(metadataBefore, file, image?.naturalWidth, image?.naturalHeight, tags);
        metadataStatus.textContent = isHeicFile ? 'Original HEIC metadata · read locally' : 'Read locally on this device';
      }
    });
    faceCount.hidden = true;
    comparisonControl.hidden = true;
    faceMarkers.replaceChildren();
    revealedFaces.replaceChildren();
    editEmpty.hidden = false;
    detectButton.disabled = false;
    exportButton.disabled = true;
    drawComparison();
    previewCaption.textContent = 'ORIGINAL · ANONYMIZED';
    setStatus('Photo loaded. Detect faces to anonymize it.');
  };
  nextImage.onerror = () => {
    URL.revokeObjectURL(objectUrl);
    setStatus('This format could not be decoded here. Try JPEG, PNG, WebP, or HEIC.');
  };
  nextImage.src = objectUrl;
}

async function createSession(): Promise<ort.InferenceSession> {
  ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, window.location.href).href;
  const modelUrl = new URL(`${import.meta.env.BASE_URL}centerface.onnx`, window.location.href).href;
  // Use the same CPU provider for photos on every device.
  ort.env.wasm.numThreads = 1;
  return ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
}

function getSession() {
  sessionPromise ??= createSession().catch((error: unknown) => {
    sessionPromise = undefined;
    throw error;
  });
  return sessionPromise;
}

function getInputSize(session: ort.InferenceSession, sourceWidth: number, sourceHeight: number) {
  const dims = session.inputMetadata[0];
  if (dims.isTensor !== true || dims.shape.length !== 4) throw new Error('The model input shape is not compatible with CenterFace.');
  const shape = dims.shape;
  const batchSize = typeof shape[0] === 'number' ? shape[0] : 1;
  const fixedHeight = typeof shape[2] === 'number' ? shape[2] : undefined;
  const fixedWidth = typeof shape[3] === 'number' ? shape[3] : undefined;
  if (fixedWidth && fixedHeight) return { width: fixedWidth, height: fixedHeight, batchSize };
  const limit = 1600;
  const factor = Math.min(1, limit / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.ceil((sourceWidth * factor) / 32) * 32,
    height: Math.ceil((sourceHeight * factor) / 32) * 32,
    batchSize,
  };
}

function makeInput(imageElement: CanvasImageSource, sourceWidth: number, sourceHeight: number, width: number, height: number, batchSize: number) {
  const source = document.createElement('canvas');
  source.width = width;
  source.height = height;
  const sourceContext = source.getContext('2d', { willReadFrequently: true })!;
  sourceContext.drawImage(imageElement, 0, 0, sourceWidth, sourceHeight, 0, 0, width, height);
  const rgba = sourceContext.getImageData(0, 0, width, height).data;
  const imageTensor = new Float32Array(3 * width * height);
  const plane = width * height;
  for (let pixel = 0; pixel < plane; pixel++) {
    imageTensor[pixel] = rgba[pixel * 4];
    imageTensor[plane + pixel] = rgba[pixel * 4 + 1];
    imageTensor[plane * 2 + pixel] = rgba[pixel * 4 + 2];
  }
  const batchTensor = new Float32Array(batchSize * imageTensor.length);
  for (let batch = 0; batch < batchSize; batch++) batchTensor.set(imageTensor, batch * imageTensor.length);
  return new ort.Tensor('float32', batchTensor, [batchSize, 3, height, width]);
}

function decode(outputs: Record<string, ort.Tensor>, inputWidth: number, inputHeight: number, sourceWidth: number, sourceHeight: number): Face[] {
  const heat = outputs['537'];
  const scale = outputs['538'];
  const offset = outputs['539'];
  if (!heat || !scale || !offset) throw new Error('The model did not return the expected CenterFace outputs.');
  const heatShape = heat.dims;
  const rows = heatShape[heatShape.length - 2];
  const columns = heatShape[heatShape.length - 1];
  const heatData = heat.data as Float32Array;
  const scaleData = scale.data as Float32Array;
  const offsetData = offset.data as Float32Array;
  const candidates: Face[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      const score = heatData[index];
      if (score <= detectionThreshold) continue;
      const width = Math.exp(scaleData[planeIndex(1, rows, columns, row, column)]) * 4;
      const height = Math.exp(scaleData[planeIndex(0, rows, columns, row, column)]) * 4;
      const oy = offsetData[planeIndex(0, rows, columns, row, column)];
      const ox = offsetData[planeIndex(1, rows, columns, row, column)];
      const x1 = Math.max(0, Math.min(inputWidth, (column + ox + 0.5) * 4 - width / 2));
      const y1 = Math.max(0, Math.min(inputHeight, (row + oy + 0.5) * 4 - height / 2));
      candidates.push({
        x1: x1 / inputWidth * sourceWidth,
        y1: y1 / inputHeight * sourceHeight,
        x2: Math.min(inputWidth, x1 + width) / inputWidth * sourceWidth,
        y2: Math.min(inputHeight, y1 + height) / inputHeight * sourceHeight,
        score,
      });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept: Face[] = [];
  for (const candidate of candidates) {
    if (kept.every((face) => intersectionOverUnion(face, candidate) < 0.3)) kept.push(candidate);
  }
  return kept;
}

function planeIndex(channel: number, rows: number, columns: number, row: number, column: number) {
  return channel * rows * columns + row * columns + column;
}

function intersectionOverUnion(a: Face, b: Face) {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = width * height;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / Math.max(1, areaA + areaB - intersection);
}

async function detectSource(source: CanvasImageSource, sourceWidth: number, sourceHeight: number): Promise<DetectionResult> {
  const session = await getSession();
  const { width, height, batchSize } = getInputSize(session, sourceWidth, sourceHeight);
  const tensor = makeInput(source, sourceWidth, sourceHeight, width, height, batchSize);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  return { boxes: decode(output, width, height, sourceWidth, sourceHeight), provider };
}

async function detectFaces(): Promise<DetectionResult> {
  if (!image) throw new Error('Choose a photo before running face detection.');
  return detectSource(image, image.naturalWidth, image.naturalHeight);
}

function processPhoto() {
  if (!image) return;
  const output = document.createElement('canvas');
  output.width = image.naturalWidth;
  output.height = image.naturalHeight;
  const outputContext = output.getContext('2d')!;
  outputContext.drawImage(image, 0, 0);
  const mode = document.querySelector<HTMLInputElement>('input[name="mode"]:checked')?.value;
  for (const face of detections) {
    if (face.blur === false) continue;
    const { x, y, width, height } = getFaceOval(face, output.width, output.height);
    if (width <= 1 || height <= 1) continue;
    if (mode === 'pixelate') {
      const scale = Math.max(1, Math.floor(Math.min(width, height) / 5));
      const tiny = document.createElement('canvas');
      tiny.width = Math.max(1, Math.round(width / scale));
      tiny.height = Math.max(1, Math.round(height / scale));
      const tinyContext = tiny.getContext('2d')!;
      tinyContext.drawImage(image, x, y, width, height, 0, 0, tiny.width, tiny.height);
      outputContext.save();
      outputContext.beginPath();
      outputContext.ellipse(x + width / 2, y + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
      outputContext.clip();
      outputContext.imageSmoothingEnabled = false;
      outputContext.drawImage(tiny, 0, 0, tiny.width, tiny.height, x, y, width, height);
      outputContext.imageSmoothingEnabled = true;
      outputContext.restore();
    } else {
      const blurRadius = Math.min(32, Math.max(8, Math.min(width, height) * 0.2));
      const padding = Math.ceil(blurRadius * 2);
      const blur = document.createElement('canvas');
      blur.width = Math.max(1, Math.ceil(width) + padding * 2);
      blur.height = Math.max(1, Math.ceil(height) + padding * 2);
      const blurContext = blur.getContext('2d')!;
      if (canvasBlurSupported) {
        blurContext.filter = `blur(${blurRadius}px)`;
        blurContext.drawImage(image, x, y, width, height, padding, padding, width, height);
      } else {
        // Safari on iOS may ignore CanvasRenderingContext2D.filter. Reduce and
        // smoothly enlarge the patch so faces are still obscured in preview/export.
        const reduction = 12;
        const reduced = document.createElement('canvas');
        reduced.width = Math.max(1, Math.ceil(width / reduction));
        reduced.height = Math.max(1, Math.ceil(height / reduction));
        const reducedContext = reduced.getContext('2d')!;
        reducedContext.imageSmoothingEnabled = true;
        reducedContext.imageSmoothingQuality = 'high';
        reducedContext.drawImage(image, x, y, width, height, 0, 0, reduced.width, reduced.height);
        blurContext.imageSmoothingEnabled = true;
        blurContext.imageSmoothingQuality = 'high';
        blurContext.drawImage(reduced, 0, 0, reduced.width, reduced.height, padding, padding, width, height);
        // The softened fallback can look washed out on iOS. Add a subtle tint
        // to the actual output pixels so preview and exported PNG stay aligned.
        blurContext.fillStyle = 'rgba(32, 37, 35, 0.22)';
        blurContext.fillRect(padding, padding, width, height);
      }
      outputContext.save();
      outputContext.beginPath();
      outputContext.ellipse(x + width / 2, y + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
      outputContext.clip();
      outputContext.drawImage(blur, x - padding, y - padding);
      outputContext.restore();
    }
  }
  processed = output;
  comparisonControl.hidden = false;
  comparisonPercent = Number(comparisonRange.value);
  comparisonDivider.style.left = `${comparisonPercent}%`;
  syncComparisonBounds();
  drawComparison();
  renderFaceMarkers();
  renderRevealedFaces();
  exportButton.disabled = false;
  if (sourceFile) {
    const generation = metadataGeneration;
    metadataStatus.textContent = 'Inspecting the anonymized PNG…';
    output.toBlob((blob) => {
      if (!blob || generation !== metadataGeneration) return;
      const exportedFile = { name: `${filename}_anonymized.png`, type: 'image/png', size: blob.size };
      void readMetadata(blob).then((tags) => {
        if (generation !== metadataGeneration) return;
        showMetadata(metadataAfter, exportedFile, output.width, output.height, tags,
          'No selected EXIF, GPS, or camera fields found in the exported PNG.');
        metadataStatus.textContent = 'Before / after inspection complete · all local';
      });
    }, 'image/png');
  }
}

function getFaceOval(face: Face, width: number, height: number) {
  const boxWidth = face.x2 - face.x1;
  const boxHeight = face.y2 - face.y1;
  const x = Math.max(0, face.x1 - boxWidth * 0.15);
  const y = Math.max(0, face.y1 - boxHeight * 0.15);
  const right = Math.min(width, face.x2 + boxWidth * 0.15);
  const bottom = Math.min(height, face.y2 + boxHeight * 0.15);
  return { x, y, width: right - x, height: bottom - y };
}

function renderFaceMarkers() {
  faceMarkers.replaceChildren();
  if (!image || !processed) return;
  detections.forEach((face, index) => {
    const oval = getFaceOval(face, canvas.width, canvas.height);
    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = `face-marker${face.blur === false ? ' is-revealed' : ''}`;
    marker.style.left = `${((oval.x + oval.width / 2) / canvas.width) * 100}%`;
    marker.style.top = `${((oval.y + oval.height / 2) / canvas.height) * 100}%`;
    marker.style.width = `${(oval.width / canvas.width) * 100}%`;
    marker.style.height = `${(oval.height / canvas.height) * 100}%`;
    marker.setAttribute('aria-label', `Face ${index + 1}: ${face.blur === false ? 'unblurred, click to blur' : 'blurred, click to leave visible'}`);
    marker.title = `Face ${index + 1} · ${face.blur === false ? 'unblurred' : 'blurred'}`;
    marker.addEventListener('pointerdown', (event) => event.stopPropagation());
    marker.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      face.blur = face.blur === false;
      processPhoto();
    });
    faceMarkers.append(marker);
  });
}

function renderRevealedFaces() {
  const revealed = detections
    .map((face, index) => ({ face, index }))
    .filter(({ face }) => face.blur === false);
  revealedFaces.replaceChildren();
  editEmpty.hidden = revealed.length > 0;
  for (const { face, index } of revealed) {
    const row = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = `Face ${index + 1}`;
    const blurButton = document.createElement('button');
    blurButton.type = 'button';
    blurButton.textContent = 'Blur';
    blurButton.setAttribute('aria-label', `Blur face ${index + 1} again`);
    blurButton.addEventListener('click', () => {
      face.blur = true;
      processPhoto();
    });
    row.append(label, blurButton);
    revealedFaces.append(row);
  }
}

function drawVideoFrame() {
  if (!isVideo || videoPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const sourceWidth = videoPreview.videoWidth;
  const sourceHeight = videoPreview.videoHeight;
  const scaleX = canvas.width / sourceWidth;
  const scaleY = canvas.height / sourceHeight;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(videoPreview, 0, 0, canvas.width, canvas.height);
  const mode = document.querySelector<HTMLInputElement>('input[name="mode"]:checked')?.value;
  if (mode !== 'pixelate' && detections.length && videoBlurContext) {
    if (videoBlurCanvas.width !== canvas.width || videoBlurCanvas.height !== canvas.height) {
      videoBlurCanvas.width = canvas.width;
      videoBlurCanvas.height = canvas.height;
      videoBlurContext = videoBlurCanvas.getContext('2d');
    }
    if (videoBlurContext) {
      videoBlurContext.clearRect(0, 0, canvas.width, canvas.height);
      videoBlurContext.filter = `blur(${Math.max(18, canvas.width / 45)}px)`;
      videoBlurContext.drawImage(videoPreview, 0, 0, canvas.width, canvas.height);
      videoBlurContext.filter = 'none';
    }
  }
  for (const face of detections) {
    const centerX = (face.x1 + face.x2) / 2 * scaleX;
    const centerY = (face.y1 + face.y2) / 2 * scaleY;
    const radiusX = (face.x2 - face.x1) * 0.68 * scaleX;
    const radiusY = (face.y2 - face.y1) * 0.72 * scaleY;
    const x = Math.max(0, centerX - radiusX);
    const y = Math.max(0, centerY - radiusY);
    const width = Math.min(canvas.width - x, radiusX * 2);
    const height = Math.min(canvas.height - y, radiusY * 2);
    if (width <= 1 || height <= 1) continue;
    if (mode === 'pixelate') {
      const block = Math.max(2, Math.floor(Math.min(width, height) / 10));
      const small = document.createElement('canvas');
      small.width = Math.max(1, Math.floor(width / block));
      small.height = Math.max(1, Math.floor(height / block));
      const smallContext = small.getContext('2d')!;
      smallContext.drawImage(canvas, x, y, width, height, 0, 0, small.width, small.height);
      context.imageSmoothingEnabled = false;
      context.drawImage(small, 0, 0, small.width, small.height, x, y, width, height);
      context.imageSmoothingEnabled = true;
    } else {
      context.save();
      context.beginPath();
      context.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
      context.clip();
      if (videoBlurContext) context.drawImage(videoBlurCanvas, 0, 0);
      context.restore();
    }
  }
}

async function processVideo() {
  if (!videoUrl) return;
  if (!('MediaRecorder' in window) || !canvas.captureStream) {
    throw new Error('Video recording is not supported in this browser. Try a recent desktop browser.');
  }
  videoChunks = [];
  detections = [];
  const canvasStream = canvas.captureStream(24);
  const capture = (videoPreview as HTMLVideoElement & { captureStream?: () => MediaStream; mozCaptureStream?: () => MediaStream });
  let audioStream: MediaStream | undefined;
  if ('AudioContext' in window && (capture.captureStream || capture.mozCaptureStream)) {
    videoPreview.muted = false;
    videoAudioContext ??= new AudioContext();
    videoAudioDestination ??= videoAudioContext.createMediaStreamDestination();
    videoAudioSource ??= videoAudioContext.createMediaElementSource(videoPreview);
    videoAudioSource.connect(videoAudioDestination);
    await videoAudioContext.resume();
    audioStream = videoAudioDestination.stream;
  } else {
    audioStream = capture.captureStream?.() ?? capture.mozCaptureStream?.();
  }
  if (audioStream) {
    for (const track of audioStream.getAudioTracks()) canvasStream.addTrack(track);
  }
  const mimeType = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
    .find((type) => MediaRecorder.isTypeSupported(type));
  if (!mimeType) throw new Error('This browser cannot export WebM video.');
  videoRecorder = new MediaRecorder(canvasStream, { mimeType });
  videoRecorder.ondataavailable = (event) => { if (event.data.size) videoChunks.push(event.data); };
  videoRecorder.onstop = () => {
    cancelAnimationFrame(videoFrame);
    window.clearInterval(videoDetectInterval);
    videoDetector?.terminate();
    videoDetector = undefined;
    videoBusy = false;
    canvasStream.getTracks().forEach((track) => track.stop());
    if (videoFailureMessage) videoChunks = [];
    exportButton.disabled = videoChunks.length === 0;
    detectButton.disabled = false;
    detectButton.textContent = 'Process video again ↗';
    if (videoChunks.length) {
      const resultBlob = new Blob(videoChunks, { type: videoChunks[0].type || 'video/webm' });
      processedVideoUrl = URL.createObjectURL(resultBlob);
      processedVideo.src = processedVideoUrl;
      processedVideo.hidden = false;
      canvas.hidden = true;
      processedVideo.load();
    } else if (videoFailureMessage) {
      videoPreview.hidden = false;
      canvas.hidden = true;
    }
    setStatus(videoFailureMessage || (videoChunks.length ? 'Video processed. Play the anonymized result here, then download it.' : 'No video output was created.'));
    videoFailureMessage = '';
  };
  videoPreview.currentTime = 0;
  videoBusy = false;
  videoFailureMessage = '';
  videoDetector = new Worker(new URL('./video-detector.worker.ts', import.meta.url), { type: 'module' });
  let resolveWorkerReady!: () => void;
  let rejectWorkerReady!: (error: Error) => void;
  const workerReady = new Promise<void>((resolve, reject) => {
    resolveWorkerReady = resolve;
    rejectWorkerReady = reject;
  });
  videoDetector.onmessage = (event: MessageEvent<{ type: string; id?: number; boxes?: Face[]; provider?: string; message?: string }>) => {
    const result = event.data;
    if (result.type === 'ready') engineStatus.textContent = `Local model · ${result.provider}`;
    if (result.type === 'error' && result.id === undefined) {
      rejectWorkerReady(new Error(result.message || 'The local video model could not start.'));
    } else if (result.type === 'error') {
      console.warn('Video face detection failed.', result.message);
      if (result.id === videoRequestId) {
        videoBusy = false;
        videoFailureMessage = 'Face detection stopped unexpectedly. No video export was kept.';
        videoPreview.pause();
        if (videoRecorder?.state === 'recording') videoRecorder.stop();
      }
    }
    if (result.type === 'result' && result.id === videoRequestId) {
      detections = stabilizeVideoBoxes(detections, result.boxes ?? []);
      engineStatus.textContent = `Local model · ${result.provider}`;
      videoBusy = false;
    }
  };
  videoDetector.onerror = (event) => {
    console.warn('Video detection worker failed.', event.message);
    videoBusy = false;
    rejectWorkerReady(new Error('The local video detection worker failed to start.'));
    videoFailureMessage = 'Face detection stopped unexpectedly. No video export was kept.';
    videoPreview.pause();
    if (videoRecorder?.state === 'recording') videoRecorder.stop();
  };
  videoDetector.postMessage({
    type: 'init',
    modelUrl: new URL(`${import.meta.env.BASE_URL}centerface.onnx`, window.location.href).href,
    wasmBase: new URL(`${import.meta.env.BASE_URL}ort/`, window.location.href).href,
  });
  try {
    await workerReady;
  } catch (error) {
    videoDetector.terminate();
    videoDetector = undefined;
    canvasStream.getTracks().forEach((track) => track.stop());
    throw error;
  }
  videoPreview.hidden = true;
  processedVideo.hidden = true;
  canvas.hidden = false;
  videoRecorder.start(1000);
  await videoPreview.play();
  detectButton.disabled = true;
  exportButton.disabled = true;
  setStatus('Processing video in real time. Keep this tab open until it finishes.', true);
  const render = () => {
    if (!videoRecorder || videoRecorder.state !== 'recording') return;
    drawVideoFrame();
    if (videoPreview.ended) {
      videoRecorder.stop();
      engineStatus.textContent = 'Video ready';
      return;
    }
    videoFrame = requestAnimationFrame(render);
  };
  videoDetectInterval = window.setInterval(async () => {
    if (videoBusy || videoPreview.paused || videoPreview.ended || !videoDetector || videoPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    videoBusy = true;
    const id = ++videoRequestId;
    try {
      const frame = await createImageBitmap(videoPreview);
      if (!videoDetector) {
        frame.close();
        videoBusy = false;
        return;
      }
      videoDetector.postMessage({ type: 'detect', id, frame }, [frame]);
    } catch (error) {
      videoBusy = false;
      console.warn('Could not sample a video frame for face detection.', error);
    }
  }, 250);
  videoFrame = requestAnimationFrame(render);
}

function stabilizeVideoBoxes(previous: Face[], current: Face[]) {
  const remaining = [...previous];
  return current.map((face) => {
    let bestIndex = -1;
    let bestScore = 0;
    for (let index = 0; index < remaining.length; index++) {
      const score = intersectionOverUnion(face, remaining[index]);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    if (bestIndex < 0 || bestScore < 0.12) return face;
    const prior = remaining.splice(bestIndex, 1)[0];
    const blend = (next: number, old: number) => next * 0.72 + old * 0.28;
    return {
      x1: blend(face.x1, prior.x1), y1: blend(face.y1, prior.y1),
      x2: blend(face.x2, prior.x2), y2: blend(face.y2, prior.y2), score: face.score,
    };
  });
}

function drawComparison() {
  if (!image) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0);
  if (!processed) return;
  const split = Math.round(canvas.width * comparisonPercent / 100);
  context.save();
  context.beginPath();
  context.rect(split, 0, canvas.width - split, canvas.height);
  context.clip();
  context.drawImage(processed, 0, 0);
  context.restore();
  context.fillStyle = '#fff';
  context.fillRect(split - 1, 0, 2, canvas.height);
  context.fillStyle = '#202523';
  context.font = '600 13px Inter, system-ui, sans-serif';
  context.textBaseline = 'top';
  context.fillText('ORIGINAL', 18, 18);
  context.fillText('ANONYMIZED', Math.min(canvas.width - 120, split + 18), 18);
}

function exportPhoto() {
  if (isVideo) {
    if (!videoChunks.length) return;
    const blob = new Blob(videoChunks, { type: videoChunks[0].type || 'video/webm' });
    downloadBlob(blob, `${filename}_anonymized.webm`);
    return;
  }
  if (!processed) return;
  processed.toBlob((blob) => {
    if (!blob) {
      setStatus('The photo could not be exported.');
      return;
    }
    downloadBlob(blob, `${filename}_anonymized.png`);
  }, 'image/png');
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.style.display = 'none';
  document.body.append(link);
  link.click();
  // Safari may not start reading the blob URL until after the click handler returns.
  // Keep the URL alive long enough for WebKit's download flow to consume it.
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
    link.remove();
  }, 60_000);
}

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (file) loadImage(file);
});

dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    input.click();
  }
});

for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
  });
}
dropzone.addEventListener('drop', (event) => {
  const file = (event as DragEvent).dataTransfer?.files[0];
  if (file) loadImage(file);
});

detectButton.addEventListener('click', async () => {
  detectButton.disabled = true;
  exportButton.disabled = true;
  if (isVideo) {
    engineStatus.textContent = 'Loading local model…';
    try {
      await processVideo();
    } catch (error) {
      console.error(error);
      detectButton.disabled = false;
      engineStatus.textContent = 'Video export unavailable';
      setStatus(error instanceof Error ? error.message : String(error));
    }
    return;
  }
  engineStatus.textContent = 'Loading local model…';
  setStatus('Detecting faces…', true);
  try {
    const result = await detectFaces();
    detections = result.boxes.map((face) => ({ ...face, blur: true }));
    processPhoto();
    engineStatus.textContent = `Local model · ${result.provider}`;
    faceCount.hidden = false;
    faceCount.textContent = `${detections.length} face${detections.length === 1 ? '' : 's'} found`;
    const message = detections.length
      ? `${detections.length} face${detections.length === 1 ? '' : 's'} anonymized. Review the preview before sharing.`
      : 'No faces detected. Review the photo carefully before sharing.';
    setStatus(canvasBlurSupported ? message : `${message} Compatibility blur active.`);
  } catch (error) {
    console.error(error);
    engineStatus.textContent = 'Model unavailable';
    const detail = error instanceof Error ? error.message : String(error);
    setStatus(`Local model failed: ${detail}`);
  } finally {
    detectButton.disabled = false;
  }
});

exportButton.addEventListener('click', exportPhoto);
comparisonRange.addEventListener('input', () => {
  comparisonPercent = Number(comparisonRange.value);
  comparisonDivider.style.left = `${comparisonPercent}%`;
  faceMarkers.style.clipPath = `inset(0 0 0 ${comparisonPercent}%)`;
  comparisonRange.setAttribute('aria-valuetext', `${comparisonPercent}% original visible`);
  if (comparisonFrame) cancelAnimationFrame(comparisonFrame);
  comparisonFrame = requestAnimationFrame(() => {
    drawComparison();
    comparisonFrame = 0;
  });
});
window.addEventListener('resize', syncComparisonBounds);
document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    if (detections.length && !isVideo) {
      processPhoto();
      setStatus('Effect updated. Review anonymized areas before exporting.');
    }
  });
});

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, { updateViaCache: 'none' })
      .then((registration) => registration.update())
      .catch(console.warn);
  });
}
