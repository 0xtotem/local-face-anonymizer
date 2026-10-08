# Third-party notices

## deface / CenterFace model

The bundled `public/centerface.onnx` is the model distributed by [ORB-HD/deface](https://github.com/ORB-HD/deface), which identifies it as an unmodified copy of the optimized `centerface_bnmerged.onnx` from [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace). The deface project and its `centerface.py` are released under the MIT License. The upstream repository credits the CenterFace code to Star-Clouds, also under MIT.

## ONNX Runtime Web

This project uses `onnxruntime-web` version 1.22.0, distributed under the MIT License. Its runtime and WASM files are installed from npm and copied into the static site at build time.

## exifr

This project uses `exifr` for local metadata inspection. It is distributed under the MIT License. Source: https://github.com/MikeKovarik/exifr.

## heic-to

HEIC/HEIF decoding uses `heic-to` 1.6.5, distributed under the GNU Lesser General Public License v3.0 or later. Its source and license are available at https://github.com/hoppergee/heic-to. The project imports it as a separate, lazy-loaded module so it is downloaded only when a HEIC/HEIF file is selected. The dependency and lockfile are included to allow rebuilding or replacing that module.
