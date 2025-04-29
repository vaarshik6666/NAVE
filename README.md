# NAVE Audio Processor

NAVE (Noise Adaptive Voice Enhancement) is a real-time audio processing tool designed to enhance voice quality by reducing noise and applying dynamic adjustments using both traditional DSP and neural network techniques.

## Features

*   **Real-time Audio I/O:** Captures audio from an input device and processes it in chunks using `sounddevice`.
*   **Noise Estimation:** Employs a `QuantumNoiseEstimator` (utilizing Mel-scale filterbanks and a custom state matrix) to estimate the background noise floor.
*   **Spectral Noise Reduction:** Performs spectral subtraction based on a learned noise profile using `librosa` for STFT/iSTFT.
*   **AI-based Enhancement:** Integrates a convolutional neural network (`NAVEModel` built with PyTorch) to enhance the spectral characteristics of the audio (Note: Model effectiveness depends on training).
*   **Dynamics Processing:** Includes Automatic Gain Control (AGC) and soft limiting/compression (`GainController`) to manage output levels.
*   **Basic GUI:** Provides a simple graphical interface (built using `sounddevice`'s callback mechanism, likely intended for integration with a GUI toolkit later) to start/stop processing and select input devices.

## Installation

1.  **Prerequisites:**
    *   Python (>= 3.8 recommended)
    *   Git
    *   (Potentially) System dependencies for `sounddevice` (e.g., `portaudio` - often installed automatically or via package managers like `apt` or `brew`). Refer to the [python-sounddevice documentation](https://python-sounddevice.readthedocs.io/en/latest/installation.html) if needed.

2.  **Clone the repository:**
    ```bash
    git clone git@github.com:vaarshik6666/NAVE.git
    cd NAVE
    ```

3.  **Create and activate a virtual environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install the package and dependencies:**
    ```bash
    pip install -e .
    ```
    *(The `-e` flag installs the package in editable mode, which is convenient for development.)*

## Usage

1.  Make sure your virtual environment is activated.
2.  Run the GUI application using the installed console script:
    ```bash
    nave-gui
    ```
3.  The GUI should appear:
    *   Select your desired **Input Device** from the dropdown menu.
    *   Click **Start** to begin real-time audio processing.
    *   Monitor the **Raw Input** and **Enhanced Output** plots.
    *   Click **Stop** to end processing.
    *   *(Note: The "Save" button functionality might need implementation)*

## Dependencies

Key dependencies are managed by `setup.py` and include:
*   NumPy
*   PyTorch
*   SoundDevice
*   Librosa
*   SciPy
*   Matplotlib

## Testing

Basic tests can be run from the root directory:

1.  **Offline Processing Test:**
    ```bash
    python tests/test_enhancement.py
    ```
    *(This script generates a test signal, processes it using `GainController`, and plots the original vs. processed audio.)*

2.  **Component Interaction Test:**
    ```bash
    python test_script.py
    ```
    *(This script tests the basic interaction between different components like the estimator, processor, model, and controller.)*

<!-- Optional Sections -->

## Contributing

[Details on how to contribute, if applicable - e.g., Fork the repo, create a branch, submit a Pull Request.]

## License

[Specify License Here - e.g., MIT License, Apache 2.0. If unsure, state "License details TBD."]