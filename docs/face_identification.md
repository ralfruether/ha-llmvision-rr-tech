# Known-person identification (local face service)

`video_analyzer_pro` and `stream_analyzer_pro` can optionally send the analyzed clip
to a local face identification service before the AI provider is called. Household
members recognized by the service are marked on the frames sent to the model with a
green box and their name, and one short paragraph with the identified names is added
to the prompt. The AI provider still decides what happens in the scene (for example
inside or outside a private zone) and writes the notification.

The feature is off unless the face service is configured **and** a call sets
`identify_persons: true`. Any failure of the face service is logged as one warning
and the analysis continues exactly as without the option.

## Setup

1. Run the face service (separate project) on a host in your LAN. It must accept
   `POST /v1/identify?camera=<slug>` with the clip as `video/mp4` body and an
   `Authorization: Bearer <token>` header.
2. In Home Assistant open **Settings → Devices & services → LLM Vision → LLM Vision
   Settings → Reconfigure** and expand **Face identification (optional)**:
   - **Face service URL**, e.g. `http://192.168.1.10:8770`. `https://` works for any
     host; plain `http://` is only accepted for private/loopback IP addresses and
     `.local`/`.lan` host names. Credentials, query strings and fragments are
     rejected.
   - **Face service token**: the bearer token of the service (16–512 characters of
     `A–Z a–z 0–9 . _ ~ + / = -`). Required when a URL is set.

   Clear the URL to disable the feature. Existing installations need no migration:
   entries without these fields behave as "disabled".
3. Set `identify_persons: true` on the service call:

```yaml
action: llmvision.stream_analyzer_pro
data:
  provider: <entry id>
  image_entity: [camera.haustuer]
  duration: 10
  fps: 1
  frame_source: stream
  clip_path: /media/llmvision/clips/haustuer.mp4
  identify_persons: true
  message: Describe who is at the front door.
```

## What is sent and when

| Service | Clip that is identified | Name labels on frames |
| --- | --- | --- |
| `video_analyzer_pro` with `fps` | the local file, or the downloaded copy of an HTTP(S)/Frigate URL | yes, frame *n* of the fps output is at *n / fps* s |
| `video_analyzer_pro` without `fps` (I-frames) | same | no, prompt paragraph only |
| `stream_analyzer_pro`, `frame_source: stream` + `clip_path` | the clip written by the capturing ffmpeg | yes, raw frame *n* is at *n / rate* s (rate = `fps`, or 1/interval without `fps`) |
| `stream_analyzer_pro`, snapshot mode + `clip_path` | the recorded clip, after the capture finished | no, prompt paragraph only |
| `stream_analyzer_pro` without `clip_path` | nothing | `face_service: error:no_clip` |

- One request per clip/camera; the camera slug is the camera entity id without
  `camera.` or the video file stem (for generic names such as `clip.mp4` the parent
  folder name before the first `_<digit>`).
- In stream mode the request runs while the camera's capture lock is still held, so
  a follow-up capture of the same camera can start up to 20 seconds later (40 seconds
  for `haustuer`, see below).
- Clips larger than 64 MB are not uploaded (`error:too_large`), and only ISO-BMFF
  video files (mp4/m4v/mov with an `ftyp` header) are ever sent (`error:no_clip`
  otherwise). In snapshot mode the clip is read while the camera lock is still held,
  so a follow-up capture to the same `clip_path` cannot replace it before upload.
  The request has a total timeout of 20 seconds (40 seconds for the camera slug
  `haustuer`, whose clips take longest when several cameras upload at once;
  `CAMERA_REQUEST_TIMEOUTS` in `face_client.py`), follows no redirects and accepts
  at most 256 KiB of JSON.
- `stream_analyzer_pro` transcodes the clip to H.264 capped at 1920 px by default.
  With `record_codec: copy` the face service receives the original camera stream
  (native resolution, compressed only once, e.g. H.265), which matches the
  conditions the face service was evaluated on. Native clips are larger; keep
  `duration` short enough to stay under the 64 MB limit. Face labels keep their
  timing: the copied clip and the analyzed frames come from the same ffmpeg session
  and both start at the stream's first keyframe. If the stream stalls and ffmpeg
  has to be stopped, the copied clip keeps (and the face service receives) the part
  recorded until then.
- A label uses the face sample closest to the time the frame actually shows within
  ±0.25 s; the box is the detected face enlarged about 1.6× to cover the head. ffmpeg's
  `fps=` filter keeps the last frame of each slot, so frame n shows the scene at about
  (n + 0.5) / fps; labels are placed for that time (measured on real recordings).
- Only names matching `^[a-z][a-z .'-]{0,39}$` are used; invalid names, scores and
  boxes are dropped. Service identifiers and model versions never reach the prompt.

## Response fields

With `identify_persons: true` the service response additionally contains:

- `persons`: `[{"name": "lea", "score": 0.756}, …]`
- `face_service`: `ok`, `disabled` (not configured) or `error:<reason>` with reason
  `timeout`, `connection`, `http_<status>`, `redirect`, `invalid_response`,
  `too_large` or `no_clip` (the first failure when several clips were sent)
- `face_service_ms`: duration of the longest identification request

Appearance re-identification (shadow mode, only when the face service runs with
`[reid] enabled = true`). These fields are **reported only**: appearance names never
reach `persons`, the prompt or the frame labels, and they must not be used to silence or
trigger alarms until the shadow evaluation is finished:

- `appearance_persons`: `[{"name", "score", "tier", "qualifies", "seed_face_score",
  "seed_age_s", "same_camera"}, …]`, the best appearance match per name over all clips.
  The service names a person track without a usable face this way when its appearance
  (mostly clothing) closely matches a person the face service recognized shortly before.
  `tier` is `strong` when that earlier face score was >= 0.50, else `weak`; `qualifies`
  is true for a strong match at most one hour old (the client enforces both limits itself,
  whatever the service reports). Names in `persons` are never repeated.
- `person_tracks` / `unnamed_tracks`: persons seen in the clips, and those without a face
  name and without a qualifying appearance match (summed over clips). Both are `null`
  when any clip has no appearance data, so `unnamed_tracks == 0` never passes on missing
  data. Safe template check:
  `{{ ai_response.get('reid') == 'ok' and ai_response.get('unnamed_tracks') is number and ai_response.get('unnamed_tracks') == 0 }}`
- `reid`: `ok`, `skipped` (the face service ran out of time for it), `error`,
  `unavailable` (the service sent no appearance data or the identification failed) or
  `disabled` (face service not configured); with several clips the first status that is
  not `ok`.

With `debug_polylines: true`, every annotated frame in `debug` also lists the drawn
`faces` (`name` and pixel `box`).

## Images and storage

- The exposed key frame (`expose_images`, timeline, notifications) is always a clean
  copy without name labels.
- Frames written to `storage_path` never contain name labels (polylines are kept as
  before).
- With `debug_polylines: true`, the annotated frames, including name labels, are
  written to `/media/llmvision/snapshots/debug-*.jpg` just like the polyline debug
  frames. Delete them when no longer needed.
- `llmvision.store_analysis_record` writes an `analysis.json` into a directory below
  `/media/llmvision` (for example the `storage_path` evidence directory), so the
  automation's decision and the complete analyzer response (LLM result, `persons`,
  appearance fields) are kept next to the clip and snapshots for later analysis:

  ```yaml
  - action: llmvision.store_analysis_record
    continue_on_error: true
    data:
      directory: "{{ evidence_directory }}"
      record: "{{ {'notification_level': notification_level, 'ai_response': ai_response} | to_json }}"
  ```

  `record` is a JSON object or a JSON string of one (at most 4 MB); the file is
  replaced atomically. Paths outside `/media/llmvision` are rejected. The record
  contains names and the model's descriptions of people: it is personal data and is
  kept as long as the evidence directory.

## Privacy and security

- **Biometric data.** Face identification processes biometric data of everyone in
  the clip. Make sure everyone living in or visiting the home is informed and that
  your use complies with local law.
- **Names reach the cloud.** Identified names are drawn into the images and added to
  the prompt, so they are sent to the configured AI provider (and kept according to
  its data policy). With debug logging enabled, LLM Vision's provider code logs the
  request text, which then includes the names.
- **Not for security actions.** The face service has no liveness check; a printed
  photo or a screen can be recognized as a household member. Never use the
  identities to unlock doors, disarm alarms or trigger other security actions.
- **Appearance matches are not identities.** They rest mostly on clothing: someone
  dressed like a household member, or a person seen right after a wrongly recognized
  face, can be named. They stay out of the prompt, the labels and the cloud, and are
  reported for evaluation only. Whole-body matching is personal data too; inform the
  household and regular visitors.
- **Plain HTTP in the LAN.** With an `http://` URL the clip and the bearer token are
  sent unencrypted; anyone able to read LAN traffic can capture them. Prefer
  `https://` or an isolated network segment. The token is stored in the Home
  Assistant config entry (like provider API keys) and is never logged by the face
  client.
