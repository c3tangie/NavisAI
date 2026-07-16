# NavisAI
This is my personal take on training a Neural Network to perform autonomous driving tasks. The objective of this project to have a network model to perform basic lane steering, recognize trafift signs, as well as stopping for stop signs and traffic lights.

This project features two convolutional neural networks (one predicts steering wheel angle while the other detects traffic control devices and regulates speed of vehicle with throttle and brake response), and is conducted on the CARLA simulator.

## NavisSteer checkpoint workflow

Colab automatically saves the best steering checkpoint to
`MyDrive/NavisAI/models/navissteer_best.pt`. Download that file into the local
`models` folder, then verify it with:

```powershell
python verify_checkpoint.py models/navissteer_best.pt
```

After the verifier prints `CHECKPOINT READY`, launch CARLA with model steering
and keyboard throttle using the VS Code configuration named
`CARLA with NavisSteer (manual throttle)`, or run:

```powershell
python manual_control.py --navissteer-model models/navissteer_best.pt `
    --map Town04 --weather ClearNoon --filter vehicle.tesla.model3
```

NavisSteer receives raw frames from its own rigid forward-facing CARLA camera;
it does not read the desktop or press keyboard keys. The predicted continuous
value is written directly to `VehicleControl.steer`. Press `J` while driving to
toggle between model steering and manual A/D steering. The display camera can
be changed independently with `TAB` without changing the model input.

The current Colab pipeline trains on RGB images from
`roicropped/roicropped` at
220x110. Its checkpoint records `input_mode=roi`, the model input size, and the
crop coordinates. Local inference reads that metadata automatically: it resizes
the dedicated front-camera frame to 220x220, takes the lower 110 rows, and sends
the resulting RGB tensor to NavisSteer. Older full-frame checkpoints remain
supported and continue to use their saved `image_size` without an ROI crop.

The LinearSteer baseline uses the same dedicated CARLA camera, ROI crop, and
continuous `VehicleControl.steer` integration. Verify and launch it with:

```powershell
python verify_checkpoint.py models/linearsteer_best.pt
python manual_control.py --linearsteer-model models/linearsteer_best.pt `
    --map Town04 --weather ClearNoon --filter vehicle.tesla.model3
```

Only one steering checkpoint may be selected at a time. In VS Code, choose
`CARLA with LinearSteer (manual throttle)` from the Run and Debug dropdown.

For a later low-speed autonomous test, add `--model-throttle 0.20`. Start with
manual throttle first so that you can release `W` immediately if steering is
unsafe. When fixed throttle is enabled, `S`/Down and Space still override it
for braking.

## CARLA map previews

To visually compare the installed CARLA maps, generate a local screenshot
gallery:

```powershell
python preview_carla_maps.py
```

The script saves road-level screenshots and an HTML gallery to
`map_previews/index.html`. To only preview a few candidates, run:

```powershell
python preview_carla_maps.py --maps Town01 Town02 Town04 Town05
```
