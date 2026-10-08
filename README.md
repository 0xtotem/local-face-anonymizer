# Covered Faces — local anonymizer prototype

A static, installable web app for blurring or pixelating faces in photos. Face detection runs in the browser using the bundled CenterFace ONNX model and ONNX Runtime Web. Images are not uploaded. Video processing is temporarily disabled because preview and export quality are not reliable enough.

## Run locally

Requirements: Node.js 20 or later.

```sh
npm install
npm run dev
```

To build and preview the static site:

```sh
npm run build
npm run preview
```

`dist/` is the deployable site. Serve it over HTTPS (or use localhost) for PWA installation and WebGPU. The model and ONNX runtime assets are bundled and served locally; the app does not fetch them from a CDN.

## Deploy to GitHub Pages

The `.github/workflows/deploy-pages.yml` workflow builds and deploys on every push to `main`, or manually from the Actions tab.

1. Push this project to a public GitHub repository on the `main` branch.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions** as the source.
3. After the **Deploy to GitHub Pages** workflow succeeds, open the URL shown in **Settings → Pages**.

The site is static; photos and videos are processed on the device and are not uploaded. The first visit downloads the model and runtime from GitHub Pages. GitHub Pages has usage limits and a soft monthly bandwidth limit; see [GitHub's Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

## What metadata is removed?

Photo export is a newly encoded PNG generated from canvas pixels. It does not copy the source file's EXIF, GPS/location, camera make/model, capture date/time, orientation tag, embedded thumbnail, or XMP/IPTC metadata. The exported image still contains its pixel dimensions and visual contents. Re-encoding does not detect or remove information visibly present in the picture, nor does it guarantee removal of steganographic or application-specific data outside standard image metadata.

The interface shows selected source-file properties and common EXIF/GPS/camera fields before processing, then checks the generated PNG for those fields after processing. Metadata is read locally. The report is a useful inspection, not a guarantee that every proprietary or nonstandard field is understood.

JPEG, PNG, WebP, and browser-native image formats can be opened when the browser supports decoding them. HEIC/HEIF files are decoded locally with `heic-to` if the browser does not natively open them; the HEIC decoder is lazy-loaded on demand. RAW camera formats and unusual containers are not supported reliably. See `THIRD_PARTY_NOTICES.md` for dependency licenses.

Video processing is currently disabled and no video export is available.

## Video support and limitations

Video support is intentionally turned off until preview, face positioning, and export can be made reliable across browsers. The unused processing prototype is not exposed by the interface.

## Model details

The app uses a dynamically shaped version of the CenterFace ONNX model distributed by `deface`; the original is preserved at `work/centerface-source.onnx`. The conversion follows `deface`'s input/output names and symbolic dimensions. To regenerate the model, install `requirements-model.txt` and run `python scripts/dynamicize_model.py`.

The photo detector limits its input's longest side to 1600 pixels and rounds dimensions to a multiple of 32. Input is RGB NCHW float32. Its confidence threshold is 0.2. Photos always use ONNX Runtime Web's WASM provider so detection does not switch between WebGPU and WASM across devices.

The CenterFace model and upstream Python dependencies are licensed under MIT according to `deface`; see `THIRD_PARTY_NOTICES.md`. ONNX Runtime Web is pinned to 1.22.0. Runtime WASM files are copied from `node_modules` into `public/ort` before dev/build.
