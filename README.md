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

Advanced face landmark detection and facial expression analysis module using MediaPipe and YARP. Detects 478 facial landmarks in real-time and extracts 52 blendshapes (facial expressions) for animation, emotion analysis, and detection of complex affective states. Includes valence-arousal dimensional emotion representation and sophisticated affective state detection algorithms.

<div align="center">
  <img src=".media/gif_landmarks.gif" alt="Face Blendshapes Demo"/>
</div>
<br>

# 🔥 Features

- **478 Facial Landmarks Detection** - High-precision face mesh tracking with real-time performance
- **52 Blendshapes Extraction** - Real-time facial expression coefficients for animation and analysis
- **Valence-Arousal Mapping** - Convert categorical emotions to 2D dimensional space (V-A)
- **Complex Affective State Detection** - Detect sophisticated emotional states:
  - Stress
  - Anxiety
  - Deep Sadness
  - Apathy
- **Multi-Modal Emotion Analysis** - Combines facial blendshapes with physiological indicators
- **Intermodal Incongruence Detection** - Identifies mismatches between facial and physiological signals
- **YARP Integration** - Seamless communication with robotics systems
- **Configurable Output** - Choose to display landmarks, blendshapes, emotions, and/or complex states
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

| Parameter            | Type    | Description                                              | Default              |
|----------------------|---------|----------------------------------------------------------|----------------------|
| --name               | string  | module name                                              | faceBlendshapes      |
| --model_path         | string  | path to MediaPipe face_landmarker.task model             | face_landmarker.task |
| --draw_landmarks     | boolean | draw facial landmarks on output image                    | True                 |
| --top_n              | integer | number of top blendshapes to output (1-52)               | 10                   |
| --send_emotion       | boolean | enable emotion output (valence-arousal)                  | True                 |
| --send_complex_states| boolean | enable complex affective state detection                 | True                 |
| --help, -h           |         | show help message and exit                               |                      |

## Example Usage

```bash
# Run with all features enabled
python main.py --name myFaceTracker --draw_landmarks True --top_n 15 --send_emotion True --send_complex_states True

# Run without landmarks visualization, only emotion analysis
python main.py --draw_landmarks False --send_emotion True

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
| `/faceBlendshapes/emotion:o`        | Output | Sends emotion data (V-A + confidence)          |
| `/faceBlendshapes/complex_state:o`  | Output | Sends complex affective state detection        |
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

## Emotion Data (Valence-Arousal)

YARP bottle containing dimensional emotion representation:

```
(emotion_label confidence valence arousal)
```

Where:
- **emotion_label** (string): Detected emotion category
- **confidence** (float): Confidence score (0.0 - 1.0)
- **valence** (float): Pleasantness dimension (-1.0 to 1.0, where -1 = negative, +1 = positive)
- **arousal** (float): Activation level (-1.0 to 1.0, where -1 = low, +1 = high)

### Emotion Categories and V-A Mapping

| Emotion     | Valence | Arousal |
|-------------|---------|---------|
| Happiness   | 0.8     | 0.6     |
| Surprise    | 0.1     | 0.8     |
| Anger       | -0.6    | 0.7     |
| Fear        | -0.7    | 0.8     |
| Sadness     | -0.7    | -0.4    |
| Disgust     | -0.6    | 0.2     |
| Contempt    | -0.5    | 0.3     |
| Neutral     | 0.0     | 0.0     |

## Complex Affective State Detection

YARP bottle containing sophisticated affective state information:

```
(state_type confidence valence arousal arousal_derivative (factor1 score1) (factor2 score2) ...)
```

Where:
- **state_type** (string): Detected complex state (stress, anxiety, deep_sadness, apathy)
- **confidence** (float): State activation level (0.0 - 1.0)
- **valence** (float): Dimensional valence (-1.0 to 1.0)
- **arousal** (float): Dimensional arousal (-1.0 to 1.0)
- **arousal_derivative** (float): Rate of change of arousal over time
- **contributing_factors** (list): Individual state activation scores

### Complex Affective States

**Stress**
- Characterized by negative valence and high arousal
- Indicates tension and pressure response
- Parameters: high beta (arousal emphasis), high incongruence weight

**Anxiety**
- Similar to stress but with even higher arousal sensitivity
- Shows heightened emotional reactivity
- Parameters: highest beta and dynamics term

**Deep Sadness**
- Characterized by very negative valence and low arousal
- Indicates prolonged negative emotional state
- Parameters: very negative alpha and beta, low incongruence weight

**Apathy**
- Neutral or positive valence with moderate arousal
- Indicates emotional regulation or inhibition
- Parameters: high gamma bias (baseline shift)

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

## 1. Real-time Expression and Emotion Analysis

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
yarp connect /faceBlendshapes/emotion:o /emotion_logger
yarpview --name /viewer
```

## 2. Monitor Specific Expressions

```bash
# Terminal: Read blendshapes
yarp read ... /faceBlendshapes/blendshapes:o

# Watch for specific expressions like smiling, blinking, etc.
```

## 3. Complex Affective State Monitoring

```bash
# Terminal: Read complex affective states
yarp read ... /faceBlendshapes/complex_state:o

# Monitor for stress, anxiety, deep sadness, or apathy detection
```

## 4. Robot Control via Facial Expressions and Emotions

Connect blendshapes and emotion outputs to robot control modules to enable robots to understand and respond to human emotional states.

# 🛠️ RPC Commands

Send commands to the module via RPC port:

```bash
# Stop/start processing
echo "process off" | yarp rpc /faceBlendshapes
echo "process on" | yarp rpc /faceBlendshapes

# Toggle landmark drawing
echo "landmarks off" | yarp rpc /faceBlendshapes
echo "landmarks on" | yarp rpc /faceBlendshapes

# Toggle emotion output
echo "emotion off" | yarp rpc /faceBlendshapes
echo "emotion on" | yarp rpc /faceBlendshapes

# Toggle complex affective state detection
echo "complex_states off" | yarp rpc /faceBlendshapes
echo "complex_states on" | yarp rpc /faceBlendshapes

# Get module status
echo "status" | yarp rpc /faceBlendshapes

# Get help
echo "help" | yarp rpc /faceBlendshapes

# Quit module
echo "quit" | yarp rpc /faceBlendshapes
```

# 📊 Performance

- **Processing Speed:** ~30 FPS on modern CPU
- **Latency:** < 50ms per frame
- **Landmarks:** 478 points per face
- **Blendshapes:** 52 expression coefficients
- **Complex State Detection:** Real-time multi-state analysis with temporal dynamics

# 🔬 Technical Details

## Valence-Arousal Conversion

The module converts categorical emotions and blendshapes to the valence-arousal dimensional space using:

1. **Categorical to V-A**: Maps emotion categories to predefined valence and arousal values
2. **Blendshape to V-A**: Extracts dimensional emotions directly from facial blendshapes:
   - Positive blendshapes (smile, cheek squint) increase valence
   - Negative blendshapes (frown, brow down) decrease valence
   - High-arousal blendshapes (wide eyes, jaw open) increase arousal
   - Low-arousal blendshapes (eye squint, mouth close) decrease arousal

## Complex Affective State Model

The module implements a mathematical framework for detecting complex affective states:

```
F_state(V, A, dA/dt, I) = sigmoid(α·V + β·A + γ + λ_I·I + λ_D·(dA/dt))
```

Where:
- **V** (Valence): Pleasantness dimension
- **A** (Arousal): Activation level
- **dA/dt** (Arousal Derivative): Rate of emotional change
- **I** (Intermodal Incongruence): Mismatch between facial and physiological signals
- **α, β, γ**: State-specific dimensional weights
- **λ_I, λ_D**: Incongruence and dynamics sensitivity parameters

## Intermodal Incongruence

Detects mismatches between facial expressions and physiological indicators:
- Negative valence with increasing arousal (inconsistency)
- Positive valence with high arousal (potential apathy or excitement)
- Can incorporate actual physiological data if available

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

## Complex states never detected

- Ensure the module has collected baseline data (first ~3 seconds)
- Adjust threshold value in `get_dominant_state()` method (default 0.55)
- Check that facial expressions match the state definitions
- Verify that arousal is changing over time (some states require dynamic changes)

# 📚 Related Work

This module implements concepts from affective computing research:
- Valence-arousal dimensional emotion model
- Facial Action Unit (AU) analysis via blendshapes
- Micro-expression detection through blendshape dynamics
- Complex affective state recognition for mental health applications

# 👥 Authors and Acknowledgment

Created for robotics and human-robot interaction research using YARP middleware with focus on affective computing.

This project is run by:
- Aislan Gabriel [agos@ecomp.poli.br](agos@ecomp.poli.br)
- Monique Tomaz [mslst@ecomp.poli.br](mslst@ecomp.poli.br)

Built with:
- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [YARP](https://www.yarp.it/) by IIT
- [Facial Blendshape Reference](https://pooyadeperson.com/the-ultimate-guide-to-creating-arkits-52-facial-blendshapes/)

# 📄 License

Released under GNU General Public License v3.0.

# 🔗 Related Projects

- [Face ID Module](https://gitlab.iit.it/cognitiveInteraction/faceID) - Face detection and identification
- [YARP](https://github.com/robotology/yarp) - Robotics middleware
- [iCub](https://icub.iit.it/) - Humanoid robot platform
- [MediaPipe](https://github.com/google/mediapipe) - Cross-platform ML framework

---

**Have fun tracking faces and detecting emotions!** 🎭✨

For questions or issues, please open an issue on GitHub.