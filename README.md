# NavisAI
This is my personal take on training a Neural Network to perform autonomous driving tasks. The objective of this project to have a network model to perform basic lane steering, recognize trafift signs, as well as stopping for stop signs and traffic lights.

## Linear steering baseline

`Baseline.py` trains a linear regression model on flattened, resized grayscale front-camera images. It reports validation Mean Squared Error (MSE) and Mean Absolute Error (MAE), then saves the best checkpoint.

The input CSV must contain one image-path column and one continuous steering-value column. Common names such as `image`, `filename`, `front`, `steering`, and `steering_angle` are detected automatically.

```powershell
python -m pip install -r requirements.txt
python Baseline.py --csv data/raw/steering/labels.csv --image-root data/raw/steering/images
```

If the dataset uses different column names, specify them explicitly:

```powershell
python Baseline.py --csv path/to/labels.csv --image-column image_name --steering-column steering_value --image-root path/to/images
```

The best model is written to `models/linear_steering.pt`, with metrics in `models/linear_steering.json`. Use `python Baseline.py --help` to see training options.
