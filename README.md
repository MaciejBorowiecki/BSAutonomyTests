# Bluespark Autonomy tests

As of today, this repository's purpose is to prepare and document current and past real-world tests

## Usage

#### Step 1:  Calibrate
   1. In `compute_focal_length.py` set `KNOWN_WIDTH` to real width in cm of the object which will be used to calibrate the camera.
   2. Place an object of known width ($W$) at a fixed, known distance ($Z$) from the camera.
   3. Run `compute_focal_length.py` and press `c` when the object is placed correctly *(in front of the camera, and step 2)*
   4. Copy the focal length value from the terminal and set `FX_PX` in `run_test.py` to this value.

### Requiremetns
    python3 -m venv venv
    source venv/bin/acitvate
    pip install requirements.txt

### Results
    images are saved in `runs/model_name/run_id