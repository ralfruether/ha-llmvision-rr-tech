# Changelog

## 1.7.2.13 - 2026-09-26

### Fixed

- Fixed `video_analyzer_pro` sending the selected key frame to the model without
  the private-zone polylines. Every analyzed frame now carries the polylines,
  including the frames written to `storage_path` and the `debug_polylines`
  snapshots and debug information. The exposed key frame remains a clean copy.

## 1.7.2.12 - 2026-09-26

### Added

- Added per-camera capture serialization for `stream_analyzer_pro`, allowing a
  new recording to start while an earlier request is still being analyzed.
- Added optional `coalesce_while_recording` and `motion_entity` service fields.
  Coalescing retains at most one follow-up capture per camera and rechecks motion
  immediately before the follow-up starts.
- Added explicit `capture.status` metadata for completed, coalesced, and skipped
  coalescing requests.

### Fixed

- Prevented overlapping `stream_analyzer_pro` recordings for the same camera
  without serializing unrelated cameras.
- Ensured capture locks and pending follow-up state are released after failures
  and cancellation, including background clip finalization.
- Prevented duplicate camera entities in one request from opening overlapping
  captures.
