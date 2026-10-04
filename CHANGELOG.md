# Changelog

## 1.7.2.18 - 2026-10-04

### Fixed

- Fixed `record_codec: copy` losing the whole clip when the camera stream stalls.
  A stalled RTSP stream keeps the connection open, so ffmpeg ignores SIGTERM, is
  killed after the stop timeout, and a regular mp4 has no index then; the clip was
  deleted and the face service got `error:no_clip`. Copy clips are now written as
  fragmented mp4 (a fragment at every keyframe and at least every second), so a
  killed recording keeps everything up to the stall. After recording, every copy
  clip is rewritten container-only into a regular `+faststart` mp4 (H.265 tagged
  `hvc1` as before); if that fails, the playable fragmented clip is kept. A killed
  clip without any fragment is still removed, and cancelled captures still delete
  their clip. The `h264` transcode is unchanged. Verified with real ffmpeg against
  a stalled RTSP stream (mediamtx with a frozen publisher): before, 22 MB without an
  index; now a playable clip of the 5 s before the stall.

## 1.7.2.17 - 2026-10-04

### Added

- Added `record_codec` to `stream_analyzer_pro`. `h264` (default) keeps the current
  phone-friendly H.264 transcode. `copy` stores the original camera stream without
  re-encoding: native resolution, compressed only once, little CPU on the Home
  Assistant host. The local face service then gets the same full-resolution input it
  was evaluated on. Stream-copied H.265 clips are retagged from `hev1` to `hvc1` (container
  only) so iPhones play them. `record_fps` and `record_scale` (other than 0) are
  rejected with `copy`. Face labels keep their timing: the copied clip and the
  analyzed frames come from the same ffmpeg session and start at the same keyframe.

### Changed

- `record_scale` no longer pre-fills 1920 in the service UI (an empty value still means
  1920 for `h264`), so ticking the field does not break `record_codec: copy`.

## 1.7.2.16 - 2026-10-02

### Fixed

- Fixed face-service name labels being drawn about half a second behind the person
  on `stream_analyzer_pro` and `video_analyzer_pro` frames. ffmpeg's `fps` filter
  keeps the last frame of each output slot, so frame n shows the scene at about
  (n + 0.5) / fps, not n / fps (measured +0.47 s at fps 1 on real recordings).
  Labels now use that time and only face samples within ±0.25 s, so a label is
  no longer borrowed from a moment the person has already moved on.

## 1.7.2.15 - 2026-10-02

### Added

- Added optional known-person identification for `video_analyzer_pro` and
  `stream_analyzer_pro` via a local face service (`identify_persons`). New optional
  Settings fields `face_service_url` and `face_service_token`. Recognized household
  members are marked with a green name label on the frames sent to the model (the
  exposed key frame stays clean) and named in one prompt paragraph; responses gain
  `persons`, `face_service` and `face_service_ms`. Failures never stop the analysis.
  See [docs/face_identification.md](docs/face_identification.md), including the
  privacy notes and why identities must not drive locks or alarms.

## 1.7.2.14 - 2026-09-27

### Fixed

- Fixed exposed key frames being deleted by timeline snapshot cleanup while the
  analyzer request was still running, which made notifications fail to attach
  the returned `key_frame` (`No such file or directory`). All timeline instances
  now share one pending-snapshot list and cleanup lock, and analyzers protect
  their key frame from the moment it is written until the timeline event is
  saved or the request fails.

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
