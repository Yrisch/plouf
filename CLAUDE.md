# CLAUDE.md — working on plouf

Guidance for AI coding agents (and humans) changing this code. User-facing docs are in
`README.md`; this file covers architecture, invariants and how to test.

plouf explores Phantom SPH dumps (view mode, the default) and builds camera paths
through them in a Qt editor (movie editor), rendered into movies. The column density
backend is shamrock (python module `shamrock`, from `<shamrock build>/pysham`) or, when
it does not import, a numpy SPH renderer (`render/sph.py`). Pure Python: numpy, scipy,
matplotlib, Pillow, PyQt6, optionally shamrock; pytest for the tests.

## Running

```bash
./plouf dumps...  [--path p.json] [--sycl 0:0] [--sample 400000] [--format phantom]
                  [--proxy-cache 1024] [--keyboard qwerty|azerty] [--backend auto|shamrock|numpy]
./plouf render p.json [--scale 0.5] [--frames a:b] [--mp4 out.mp4] [--sycl 0:0] [--backend ...]
./plouf test [pytest options]
```

The launcher puts `src/` and `$SHAMROCK_BUILD/pysham` (default `../../build`) on
`PYTHONPATH`, the build on `LD_LIBRARY_PATH`, and runs `$PLOUF_PYTHON` (default:
`python3` on `PATH`) with `-m plouf.gui.app`, `-m plouf.render_movie` or
`-m pytest`. Without `pysham/` it warns and runs anyway (numpy renderer). Nothing is
compiled here; shamrock, if used, must already be built (README, "Installation").
Without `--path` the window opens in the view mode, with it in the movie editor.

## Layout

| module | role | may import |
|---|---|---|
| `constants.py` | units (cgs), `SINK_MODES`, `STAR_MAX_RACC_PC` (default star / core threshold of the cluster mode) | — |
| `path/camera.py` | `CamState`, camera maths (`basis`, orbit/turntable/look/level, `_keep_hint`), `project`, `unproject_delta`, `frustum_corners`, `stereo_eyes` | numpy |
| `path/campath.py` | `SerialHermite`, `Smoothstep`, `Keyframe`, `CameraPath` (keys → state at t, following, `retime_segment`), `make_orbit`, `DEFAULT_LOOK` | path |
| `data/readers.py` | fast partial dump readers: `DumpReader`, `PhantomReader`, `READERS`, `get_reader` | numpy |
| `data/sinktracks.py` | `SinkTracks`: every sink in every dump, Hermite positions, `raw_at` / `sinks_at(dump, look)`, `split_sinks` (sink modes), npz cache in `~/.cache/plouf/` | data, constants |
| `data/proxycache.py` | `proxy_sample` (fixed-rank gas sample with grown smoothing lengths), `ProxyCache` (LRU, prefetch by reader threads, in worker processes for the editor) | data, constants |
| `render/compose.py` | gas colormap, proxy `splat_sph` (adaptive, with h) / `splat_column`, sink discs (`disc_cover`), label, `compose`, `save_png` | path, numpy/scipy/matplotlib/PIL |
| `render/stars.py` | `StarLight` (photometry, PSF, stretch, dots, auto exposure), `star_exposure` | path |
| `render/sph.py` | numpy SPH renderer: `column_render`, `column_to_points`, `NumpyScene` (`SimScene` interface), `classify_sinks` | path, data, constants, numpy/scipy |
| `render/backend.py` | `SimScene` (shamrock load + column renders), `make_scene` (backend choice, numpy fallback), `FrameRenderer`, `server_main` (render process) | shamrock, everything above |
| `gui/*.py` | Qt main window: `app.py` (`Editor`: both modes, menus, movie editor; `main`), `viewmode` (`ViewMode`, `ViewCanvas`, `ViewPanel`), `worldview`, `viewfinder`, `timeline`, `inspectors` (panels, dialogs), `workers` (render process client, tracks thread), `keymaps`, `qtutil` | everything |
| `render_movie.py` | batch renderer | path, data, render |

Imports inside the package are relative. Keep `path/`, `data/`, `render/compose.py`,
`render/stars.py`, `render/sph.py` free of Qt and of shamrock (only `render/backend.py`
imports shamrock, lazily), so they stay importable and testable without either.

## Processes and threads (editor)

* **Render process**: shamrock (or the numpy renderer) runs in a separate
  `multiprocessing` *spawn* process (`render.backend.server_main`) talking over a `Pipe`.
  Requests are dicts (`{"cmd": "render"|"tracks"|"quit", ...}`, renders carry the dump
  `format`); the server drains the pipe and only serves the latest render request (stale
  ones are dropped, except `keep` exports), and first replies `{"type": "backend", "name"}`.
  A render in progress is cancelled when the view it is for is gone: `Viewfinder`
  (`_drop_stale_render`, at paint) calls `Backend.cancel()`, which bumps a shared
  generation counter; the server's `check()` (between steps, and between chunks in the
  numpy renderer) raises `Cancelled` for older non-`keep` requests and replies
  `{"type": "cancelled", "id"}`. The process runs at `RENDER_NICE`. Never import or call
  shamrock in the GUI process.
* **Modes**: `Editor` holds a `QStackedWidget` with the view mode (`ViewMode`) and the
  movie editor page; `Editor.mode` is `"view"` or `"movie"` (`set_mode`). Only the shown
  mode asks for renders and plans the proxy prefetching (`Editor.request_refine` /
  `update_proxy` defer to the view mode or return). Both listen to the backend's `frame`
  / `error` signals and keep the replies whose request id is theirs.
* **Worker threads**: `TracksBuilder` (QThread, sink tracks) and `ProxyCache`'s reader
  threads, which in the editor only dispatch to worker processes (`processes=True`: the
  reading does not take the GUI thread's GIL time; the processes exit with their parent,
  `_exit_with_parent`). Workers never touch widgets: they report through Qt signals
  (`ProxyBridge`) back to the GUI thread; `ProxyCache.close()` stops all callbacks.
* **Movie render** from the editor: a `QProcess` running `-m plouf.render_movie`,
  with `src/` added to its `PYTHONPATH`.
* The instant views (proxy splat, world view) must stay cheap: they run on every mouse
  move and while playing. `Viewfinder.proxy_image` draws a coarse proxy (`PROXY_FAST_PIXELS`,
  half of the sample) while the view changes and the full one (`PROXY_PIXELS`) after
  `PROXY_SETTLE_MS`. `compose` without a star layer takes a uint8 fast path that must give
  the same pixels as the float path (tested). Measure before adding per-frame work there.

## Invariants — do not break (most are tested)

1. **Serial, local interpolation.** The path is a chain of cubic Hermite segments, C1 at
   keys. The velocity at key *k* only uses keys ≤ *k* (parabola through *k-2, k-1, k*);
   exceptions that look at the next key: holds, ease keys, reversals of monotone scalars.
   Consequences that must keep holding:
   * appending a key leaves every earlier segment unchanged (in the exception cases only
     the previous last segment changes);
   * editing key *k* only changes segments *k-1 … k+2*;
   * `CameraPath.evaluate` at a key time returns that key's state exactly (up as typed);
   * consecutive keys with the same position (or target, or up) hold perfectly still,
     even when their dumps differ;
   * retiming a segment (`CameraPath.retime_segment`, `Keyframe.stretch`) leaves every
     other segment unchanged, only shifted in time, with the same velocities at its ends:
     `SerialHermite` estimates velocities on the unstretched durations `h / stretch`, and
     only clamps the ends of the stretched segment itself (no overshoot past ~3×).
   Never replace this with a global spline.
2. **Position is relative to the target**: direction of `pos - target` + log distance,
   with monotone (Fritsch–Carlson) interpolation for scalars so nothing overshoots.
3. **Following**: one set of splines per reference frame (world + each followed sink),
   key values stored minus the sink position at the key dump, blended with smoothstep
   weights. Don't mix frames inside one spline (it caused distance dips before).
4. **`up` is stored as typed** (e.g. exactly `[0, 0, 1]`); only roll changes it.
   `turntable` / `look` pivot around it (`_keep_hint`).
5. **Sinks are identified by their index** in the dump sink arrays (stable in Phantom;
   dead sinks keep their slot with mass ≤ 0). Positions between dumps come from
   `SinkTracks` (Hermite with velocities) both in the editor and in renders, so a
   followed sink is exactly where the camera expects it.
6. **Shamrock side is untouched by the proxy cache.** The fast readers only feed the
   instant views and sink tracks; real renders always load dumps through shamrock.
7. **Proxy sample stability**: the living particles with the lowest `particle_rank(iorig)`;
   accreted particles are replaced by the next ranked ones so the sample keeps *n*
   particles and does not flicker. Weight = living gas mass / n (in M☉/pc² splat units).
8. **Resolution independence of stars**: PSF sizes are defined on a 1920 px frame and
   scaled by `nx / REF_WIDTH`, so drafts (`--scale`) look like the final movie.
9. **Stars without PSF look like the proxy**: `StarLight.dots` gives brightness, colour,
   radius and opacity to both the render (`_dots`, RGBA drawn over the gas by `compose`)
   and the viewfinder proxy (`_proxy_stars`), sizes relative to `DOT_REF_WIDTH`. Change
   the dot look there only.
10. **Auto exposure is a function of path time only** (`stars.star_exposure`, ±0.5 s
   window, clamped ±3 dex), so frames can be rendered in any order or in parallel.
11. **Stereo eyes are parallel with a lens shift**, never toed in: `stereo_eyes` moves the
   mono camera along its right vector (eyes `angle` apart seen from the target) and sets
   `CamState.shift` (horizontal lens shift in half image widths, a whole number of pixels,
   never saved) so the target is centred in both eyes. Everything that maps world to
   pixels goes through `project`, which applies the shift; the gas render
   (`SimScene.column`) renders a centred camera `2|shift|` px wider and crops. A new
   projection elsewhere must honour `shift` too. Both eyes use the mono star exposure.
12. **The numpy renderer integrates along the real rays.** Each pixel gets
   `m / h^2 F(d / h)` per particle, `d` the distance from the particle to that pixel's
   ray (`_stamp_q2`), not to the pixel in the image plane (that was 10-20% off near the
   frame edges). Particles below `NORM_BELOW` px spread exactly their projected mass
   (normalised stamps); the coarse levels are chosen from the footprint's width so the
   kernel stays resolved across. It must keep matching shamrock
   (`test_numpy_renderer_matches_shamrock`) and the exact kernel integrals (`test_sph.py`).
13. **The modes are independent**: the view mode has its own `CamState`, look, selection
   and preview; switching mode changes neither camera. In the view mode the movie
   editor's actions and window shortcuts are disabled (and vice versa for `Navigate`),
   so Space / Esc / Ctrl+E mean one thing at a time. View keys (`keymaps.VIEW_KEYS`) must
   not use a fly / roll key of either layout.
14. **Sinks are split by the look**, never at load: scenes, tracks and the render process
   carry the raw alive sinks (`id`, `pos`, `mass` [M☉], `racc`), and `split_sinks(look)`
   decides stars / discs per frame (`look["sinks"]`, `constants.SINK_MODES`). Old path files
   (no `sinks` key) load as `"cluster"` so they render as before; new ones default to markers.

## Extending

* **New dump format**: subclass `DumpReader` in `data/readers.py`, implement
  `read_sinks` and `read_gas` (shapes and units in the docstrings, `h` included: the
  numpy renderer renders from it), register it in `READERS`. Shamrock loading for renders
  is in `render/backend.py` `SimScene.load`, which is Phantom-only today. Add a synthetic
  dump writer next to `tests/phantom.py`.
* **New look option**: add the key with its default to `DEFAULT_LOOK` (old path files
  get it through `CameraPath.__post_init__`), a control in `gui/inspectors.py`
  `MovieInspector`, and use it in `render/compose.py` / `render/stars.py` /
  `FrameRenderer.render`. If it affects the instant preview, mirror it in
  `gui/viewfinder.py`; if the view mode should offer it, add it to `ViewPanel`.
* **New per-key value**: add it to `CamState` (with a default, so old files load), interpolate
  it in `CameraPath._build` / `_world`, expose it in `CameraInspector`. Use monotone
  mode for scalars that must not overshoot.
* **Star look**: tuning constants are at the top of `StarLight` in `render/stars.py`
  (`I_SOFT`, `I_WHITE`, `CORE_FWHM`, `HALO_*`, `SPIKE_FRAC`, `RADIUS`, extinction ratios). The look options `spike_angle`
  and `glow_gamma` (`StarLight.limit_glow`, stars brighter than white) act on top of them.
* **Keyboard**: shortcuts go through `KEYMAPS` / `key_char` in `gui/keymaps.py` (QWERTY
  and AZERTY), view mode keys through `VIEW_KEYS` / `view_shortcuts` (each one is a
  `Navigate` menu action, `ViewMode.build_menu`); never hard-code a letter.

## Testing

```bash
./plouf test                                   # ~12 s, no shamrock needed
PLOUF_TEST_DUMP=/path/to/dump ./plouf test     # + shamrock tests (render process, reader and numpy renderer vs shamrock)
./plouf test tests/test_interpolation.py -k append
```

* `tests/conftest.py` writes a small synthetic simulation (`sim` fixture): 5 Phantom
  dumps (the last one a single precision "small" dump), a gas ball losing particles to
  accretion, sinks moving on straight lines (so Hermite tracks are exact), one star
  forming at dump 2, one core. `sink_pos(s, dump)` is the exact sink position.
  `XDG_CACHE_HOME` is redirected per test.
* `tests/test_gui.py` runs the real `Editor` offscreen with `StubBackend` in place of
  the render process (`answer()` replies to a request with a blank frame); it drives it
  with Qt events (clicks, keys) and editor methods. Fixtures: `editor` (movie editor),
  `viewer` (the `ViewMode` of an editor in its default view mode). Wait for background
  work with the `wait(app, cond)` helper, never with sleeps.
* `tests/test_sph.py` checks the numpy renderer against brute-force ray integrals (no
  shamrock needed) and runs the render process and the batch renderer with it.
* Each invariant above should have a test; add one with any new behaviour, and run the
  whole suite before finishing. A change that alters rendered pixels on purpose should
  say so; otherwise compare before/after images.
* Manual GUI checks outside pytest: `QT_QPA_PLATFORM=offscreen`, `XDG_CONFIG_HOME` to a
  temp dir; the editor with no dumps opens a file dialog, and closing with unsaved changes
  opens a "save?" dialog — both hang a headless run (end scripts with `os._exit(0)`).

## Style

* Comments and docstrings explain *why* and give units; code units vs physical units
  (pc, M☉, M☉/pc²) are always explicit in names or comments. Constants go in
  `constants.py`, not as literals.
* Vectorise with numpy; avoid per-particle Python loops (per-star loops in `StarLight`
  are bounded by the visible stars).
* Match the surrounding code: short helpers, no new dependencies without need.
* When changing behaviour visible to users, update `README.md` in the same change.
* Every source file starts with the license header (4 comment lines, `SPDX-License-Identifier:
  CECILL-2.1`); copy it from any existing file into new ones.
