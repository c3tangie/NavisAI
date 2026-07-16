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
    --map Town02 --weather ClearNoon --filter vehicle.tesla.model3
```

NavisSteer receives raw frames from its own rigid forward-facing CARLA camera;
it does not read the desktop or press keyboard keys. The predicted continuous
value is written directly to `VehicleControl.steer`. Press `J` while driving to
toggle between model steering and manual A/D steering. The display camera can
be changed independently with `TAB` without changing the model input.

For a later low-speed autonomous test, add `--model-throttle 0.20`. Start with
manual throttle first so that you can release `W` immediately if steering is
unsafe. When fixed throttle is enabled, `S`/Down and Space still override it
for braking.
