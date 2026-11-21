# Phase 0.5 — Coordinate transform report

## Cause

The upload path checked source dimensions, decoded the image, then called
`enhance_image()`. That function may upscale inputs whose short side is below
480 px. Inference polygons and response width/height used the processed image,
while the browser rendered the original uploaded file. The response therefore
had no reliable mapping for those inputs.

## Change

`backend/coordinate_transform.py` now defines `ImageTransform`, recording:

- original and processed width/height;
- actual independent horizontal and vertical scale factors;
- crop offsets and padding offsets (zero for today's preprocessing, represented
  explicitly for future transforms).

`preprocess_image()` returns both the enhanced image and transform. The predict
route maps polygon vertices back to original-image pixel-edge coordinates,
clips to original bounds, recalculates mask metrics at original dimensions, and
returns original width/height. A `preprocessing` response object exposes the
transform record. No bounding box field is emitted by the current API; the
transform also provides tested xyxy mapping for internal/future callers.

Existing response keys remain. `preprocessing` and `runtime` are additive. The
two-field `read_image()` result remains the default for feedback/collector
callers; prediction opts into the transform record.

## Tests

`backend/test_coordinate_transform.py` covers unchanged-size images, upscaling,
non-square independent scales, crop/padding inversion, boundary clipping,
polygons and xyxy boxes. In the complete backend test run, these six tests
passed. API-level image rendering was not exercised because the local API
listener cannot bind in this execution environment.

## Manual verification still required

Compare the returned `preprocessing` record and displayed overlays using an
original-resolution browser upload, including a small image that is upscaled by
the 480-pixel minimum-side rule and an EXIF-rotated phone photo. The server and
browser must agree on orientation for the source coordinate system.
