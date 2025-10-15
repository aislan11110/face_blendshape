<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="100%"
        src="https://raw.githubusercontent.com/aislan11110/face_blendshape/refs/heads/main/media/gif_landmarks.gif"
      >
    </a>
  </p>

  [![license](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
  [![python-version](https://img.shields.io/badge/python-3.11-blue)]()

</div>
# 📦 Description
Face landmark detection and facial expression analysis module using MediaPipe and YARP.
Detects 478 facial landmarks in real-time and extracts blendshapes (facial expressions) for animation and emotion analysis.

<div align="center">
  <img src=".media/gif_landmarks.gif" alt="Face Blendshapes Demo"/>
</div>
<br>

# 🔥 Features

- **478 Facial Landmarks Detection** - High-precision face mesh tracking
- **52 Blendshapes Extraction** - Real-time facial expression coefficients
- **YARP Integration** - Seamless communication with robotics systems
- **Configurable Output** - Choose to display landmarks and/or blendshapes
- **Top-N Blendshapes** - Output only the most significant expressions

# 🚀 How to Run

Once installed, open YARP manager or run from terminal:
```bash
# Start YARP server
yarpserver

# Run the module
python main.py
```

The module runs with the following parameters:

| Parameter         | Type    | Description                                              | Default              |
|-------------------|---------|----------------------------------------------------------|----------------------|
| --name            | string  | module name                                              | faceBlendshapes      |
| --model_path      | string  | path to MediaPipe face_landmarker.task model             | face_landmarker.task |
| --draw_landmarks  | boolean | draw facial landmarks on output image                    | True                 |
| --top_n           | integer | number of top blendshapes to output (1-52)               | 10                   |
| --help, -h        |         | show help message and exit                               |                      |

## Example Usage
```bash
# Run with custom settings
python main.py --name myFaceTracker --draw_landmarks True --top_n 15

# Run without landmarks visualization
python main.py --draw_landmarks False

# Specify custom model path
python main.py --model_path /path/to/face_landmarker.task
```

# 🔌 YARP Ports

The module creates the following ports:

| Port Name                           | Type   | Description                                    |
|-------------------------------------|--------|------------------------------------------------|
| `/faceBlendshapes/image:i`          | Input  | Receives RGB images for processing             |
| `/faceBlendshapes/annotated_image:o`| Output | Sends images with drawn landmarks              |
| `/faceBlendshapes/blendshapes:o`    | Output | Sends blendshapes data as YARP bottles         |
| `/faceBlendshapes`                  | RPC    | Command port for module control                |

# 📝 Output Format

## Annotated Image
RGB image with 478 facial landmarks and connections drawn on detected faces.

## Blendshapes Data
YARP bottle containing the top-N blendshapes with scores (0.0 - 1.0):
```
(("eyeBlinkLeft" 0.95) ("eyeBlinkRight" 0.92) ("mouthSmile" 0.78) ...)
```

### Available Blendshapes (52 total)

**Eyes:**
- eyeBlinkLeft, eyeBlinkRight
- eyeLookDownLeft, eyeLookDownRight
- eyeLookInLeft, eyeLookInRight
- eyeLookOutLeft, eyeLookOutRight
- eyeLookUpLeft, eyeLookUpRight
- eyeSquintLeft, eyeSquintRight
- eyeWideLeft, eyeWideRight

**Eyebrows:**
- browDownLeft, browDownRight
- browInnerUp
- browOuterUpLeft, browOuterUpRight

**Mouth:**
- mouthClose
- mouthFunnel
- mouthPucker
- mouthLeft, mouthRight
- mouthSmileLeft, mouthSmileRight
- mouthFrownLeft, mouthFrownRight
- mouthDimpleLeft, mouthDimpleRight
- mouthStretchLeft, mouthStretchRight
- mouthRollLower, mouthRollUpper
- mouthShrugLower, mouthShrugUpper
- mouthPressLeft, mouthPressRight
- mouthLowerDownLeft, mouthLowerDownRight
- mouthUpperUpLeft, mouthUpperUpRight

**Jaw:**
- jawOpen
- jawForward
- jawLeft, jawRight

**Cheeks:**
- cheekPuff
- cheekSquintLeft, cheekSquintRight

**Nose:**
- noseSneerLeft, noseSneerRight

**Tongue:**
- tongueOut

# 💻 Installation

## Prerequisites
- Python 3.11
- YARP
- Conda or Miniforge

## Setup
```bash
# Create conda environment
conda create -n face_blend python=3.11 -y
conda activate face_blend

# Install YARP
conda install -c conda-forge -c robotology yarp -y

# Install Python dependencies
pip install mediapipe opencv-python numpy

# Download MediaPipe model (if not included)
# The face_landmarker.task model will be loaded from the module directory
```

# 🎯 Example Applications

## 1. Real-time Expression Analysis
```bash
# Terminal 1: YARP server
yarpserver

# Terminal 2: Webcam stream
python webcam_yarp.py

# Terminal 3: Face blendshapes module
python main.py

# Terminal 4: Connections
yarp connect /webcam /faceBlendshapes/image:i
yarp connect /faceBlendshapes/annotated_image:o /viewer
yarpview --name /viewer
```

## 2. Monitor Specific Expressions
```bash
# Terminal: Read blendshapes
yarp read ... /faceBlendshapes/blendshapes:o

# Watch for specific expressions like smiling, blinking, etc.
```

## 3. Robot Control via Facial Expressions

Connect blendshapes output to robot control modules to make robots mimic human expressions.

# 🛠️ RPC Commands

Send commands to the module via RPC port:
```bash
# Stop/start processing
echo "process off" | yarp rpc /faceBlendshapes
echo "process on" | yarp rpc /faceBlendshapes

# Toggle landmark drawing
echo "landmarks off" | yarp rpc /faceBlendshapes
echo "landmarks on" | yarp rpc /faceBlendshapes

# Quit module
echo "quit" | yarp rpc /faceBlendshapes

# Get help
echo "help" | yarp rpc /faceBlendshapes
```

# 📊 Performance

- **Processing Speed:** ~30 FPS on modern CPU
- **Latency:** < 50ms
- **Landmarks:** 478 points per face
- **Blendshapes:** 52 expression coefficients

# 🔧 Troubleshooting

## Module fails to start

**Error:** `Cannot find face_landmarker.task`

**Solution:** Ensure the MediaPipe model file is in the module directory or specify the correct path:
```bash
python main.py --model_path /full/path/to/face_landmarker.task
```

## No faces detected

- Ensure adequate lighting
- Face should be clearly visible and frontal
- Check if input image is being received: `yarp read ... /faceBlendshapes/image:i`

## Slow performance

- Disable landmark drawing: `python main.py --draw_landmarks False`
- Reduce output resolution from webcam
- Check CPU usage

# 📚 Technical Details

## MediaPipe Face Landmarker

This module uses Google's MediaPipe Face Landmarker solution:
- **Model:** face_landmarker.task (v2)
- **Input:** RGB images
- **Output:** 478 3D facial landmarks + 52 blendshapes

## Coordinate System

- Landmarks: Normalized coordinates (0.0 - 1.0)
- Blendshapes: Float values (0.0 - 1.0) representing expression intensity

# 👥 Authors and Acknowledgment

Created for robotics and human-robot interaction research using YARP middleware.

This project is run by Aislan Gabriel [agos@ecomp.poli.br](agos@ecomp.poli.br)

Built with:
- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [YARP](https://www.yarp.it/) by IIT
- More info about Facial Blendshape on [here](https://pooyadeperson.com/the-ultimate-guide-to-creating-arkits-52-facial-blendshapes/)

# 📄 License

Released under GNU General Public License v3.0.

# 🔗 Related Projects

- [Face ID Module](https://gitlab.iit.it/cognitiveInteraction/faceID) - Face detection and identification
- [YARP](https://github.com/robotology/yarp) - Robotics middleware
- [iCub](https://icub.iit.it/) - Humanoid robot platform

---

**Have fun tracking faces!** 🎭✨

For questions or issues, please open an issue on GitHub.

This README.md was generated by Claude Sonnet 4.5 modified by Aislan