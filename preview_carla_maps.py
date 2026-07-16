"""Generate visual previews for installed CARLA maps.

This script connects to CARLA, loads each selected map, captures a few
road-level RGB camera screenshots, and writes a small HTML gallery.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import queue
import subprocess
import time

import carla


def find_local_carla_server():
    """Return the packaged CARLA server executable for this workstation."""
    configured_path = os.environ.get("CARLA_SERVER_PATH")
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    directory = Path(__file__).resolve().parent
    while True:
        candidates = (
            directory / "CarlaUE4.exe",
            directory / "CARLA_0.9.16" / "CarlaUE4.exe",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        parent = directory.parent
        if parent == directory:
            break
        directory = parent

    return None


def stop_local_carla_server(server_process):
    """Stop the launcher and the Unreal child process that it created."""
    if not server_process or server_process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(server_process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        server_process.terminate()
        try:
            server_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            server_process.kill()


def connect_or_start_server(host, port, show_server_window=False):
    """Connect to CARLA, automatically starting a missing local server."""
    client = carla.Client(host, port)
    client.set_timeout(2.0)
    try:
        client.get_world()
        client.set_timeout(120.0)
        return client, None
    except RuntimeError:
        pass

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"CARLA server is not reachable at {host}:{port}")

    server_path = find_local_carla_server()
    if not server_path:
        raise RuntimeError(
            "CARLA server was not found. Set CARLA_SERVER_PATH to CarlaUE4.exe."
        )

    command = [server_path, f"-carla-rpc-port={port}"]
    if not show_server_window:
        command.extend(["-RenderOffScreen", "-nosound"])

    print("CARLA server is offline; starting it automatically...")
    server_process = subprocess.Popen(
        command,
        cwd=os.path.dirname(server_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise RuntimeError("CarlaUE4.exe exited before the server became ready.")

        client = carla.Client(host, port)
        client.set_timeout(2.0)
        try:
            client.get_world()
            client.set_timeout(120.0)
            print("CARLA server is ready.")
            return client, server_process
        except RuntimeError:
            time.sleep(1.0)

    stop_local_carla_server(server_process)
    raise RuntimeError("Timed out waiting for the CARLA server to start.")


def short_map_name(map_path):
    return map_path.rsplit("/", 1)[-1]


def sorted_town_maps(client):
    maps = [short_map_name(path) for path in client.get_available_maps()]
    maps = [name for name in maps if name.lower().startswith("town")]
    return sorted(set(maps))


def set_weather(world, weather_name):
    if not weather_name:
        return
    if not hasattr(carla.WeatherParameters, weather_name):
        raise ValueError(f"Unknown CARLA weather preset: {weather_name}")
    world.set_weather(getattr(carla.WeatherParameters, weather_name))


def representative_spawn_indices(spawn_points, count):
    if not spawn_points:
        return []
    if len(spawn_points) <= count:
        return list(range(len(spawn_points)))

    if count == 1:
        return [0]

    last_index = len(spawn_points) - 1
    return sorted(
        {
            round(index * last_index / (count - 1))
            for index in range(count)
        }
    )


def capture_camera_image(world, camera_transform, output_path, width, height, fov):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))

    image_queue = queue.Queue(maxsize=1)
    camera = world.spawn_actor(camera_bp, camera_transform)
    camera.listen(image_queue.put)

    try:
        world.wait_for_tick()
        image = image_queue.get(timeout=10.0)
        image.save_to_disk(str(output_path))
    finally:
        camera.stop()
        camera.destroy()


def preview_map(client, map_name, output_dir, shots_per_map, width, height, fov):
    print(f"Loading {map_name}...")
    world = client.load_world(map_name)
    set_weather(world, "ClearNoon")
    time.sleep(1.0)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        print(f"  skipped {map_name}: no spawn points")
        return []

    captures = []
    for shot_number, spawn_index in enumerate(
        representative_spawn_indices(spawn_points, shots_per_map),
        start=1,
    ):
        spawn = spawn_points[spawn_index]
        camera_location = spawn.location + carla.Location(z=2.2)
        camera_rotation = carla.Rotation(
            pitch=-6.0,
            yaw=spawn.rotation.yaw,
            roll=0.0,
        )
        camera_transform = carla.Transform(camera_location, camera_rotation)
        file_name = f"{map_name}_shot{shot_number}.png"
        output_path = output_dir / file_name
        capture_camera_image(
            world,
            camera_transform,
            output_path,
            width=width,
            height=height,
            fov=fov,
        )
        captures.append(file_name)
        print(f"  saved {file_name}")

    return captures


def write_gallery(output_dir, map_to_images):
    cards = []
    for map_name, images in map_to_images.items():
        image_tags = "\n".join(
            f'<a href="{html.escape(image)}"><img src="{html.escape(image)}" '
            f'alt="{html.escape(map_name)} preview"></a>'
            for image in images
        )
        cards.append(
            f"""
            <section class="map-card">
              <h2>{html.escape(map_name)}</h2>
              <div class="shots">{image_tags}</div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARLA Map Previews</title>
  <style>
    body {{
      margin: 24px;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    p {{
      color: #bbb;
      margin-top: 0;
    }}
    .map-card {{
      border-top: 1px solid #333;
      padding: 20px 0;
    }}
    .shots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    img {{
      width: 100%;
      border: 1px solid #333;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>CARLA Map Previews</h1>
  <p>Road-level screenshots captured from representative spawn points.</p>
  {''.join(cards)}
</body>
</html>
"""
    gallery_path = output_dir / "index.html"
    gallery_path.write_text(page, encoding="utf-8")
    return gallery_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate road-level screenshot previews for CARLA maps."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--output", default="map_previews")
    parser.add_argument("--maps", nargs="*", help="specific maps, e.g. Town04 Town06")
    parser.add_argument("--shots-per-map", default=3, type=int)
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=540, type=int)
    parser.add_argument("--fov", default=90, type=float)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print installed Town maps without taking screenshots",
    )
    parser.add_argument(
        "--show-server-window",
        action="store_true",
        help="show the automatically started CARLA server window",
    )
    args = parser.parse_args()

    client, server_process = connect_or_start_server(
        args.host,
        args.port,
        show_server_window=args.show_server_window,
    )
    try:
        maps = args.maps or sorted_town_maps(client)
        if args.list_only:
            print("Installed Town maps:")
            for map_name in maps:
                print(f"  {map_name}")
            return

        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        map_to_images = {}
        for map_name in maps:
            map_to_images[map_name] = preview_map(
                client,
                map_name,
                output_dir,
                shots_per_map=args.shots_per_map,
                width=args.width,
                height=args.height,
                fov=args.fov,
            )

        gallery_path = write_gallery(output_dir, map_to_images)
        print(f"Preview gallery written to: {gallery_path}")
    finally:
        stop_local_carla_server(server_process)


if __name__ == "__main__":
    main()
