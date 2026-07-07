# NAOS (NeuroAdaptive OS)

The **NAOS (NeuroAdaptive OS)** is a real-time Brain-Computer Interface (BCI) framework designed to enable hands-free human-computer interaction for assistive applications.

One of the biggest bottlenecks in BCI development is that collecting high-quality EEG data requires specialized hardware, making rapid iteration difficult. To solve this, this project includes a synthetic EEG generation pipeline called **NEST (Neural EEG Synthesis Testbed)**. NEST produces highly realistic, multi-channel EEG signals using a combination of procedural generation and GAN-based synthesis.

By streaming these simulated signals in real-time via Lab Streaming Layer (LSL) and formatting them as OpenBCI-compatible packets, NAOS creates an environment where downstream signal-processing pipelines and UI interactions can be developed and validated exactly as if they were connected to physical hardware.

Building NAOS bridges multiple domains: real-time systems, DSP (digital signal processing), machine learning, and desktop application architecture. Rather than focusing on a single algorithm, NAOS is designed as a modular framework so that new cognitive-state detectors can be plugged in independently.

## System Architecture

The framework consists of three primary components:

1. **NEST (Signal Simulator):** Generates synthetic, realistic 8-channel EEG and EOG data, broadcasting it via an LSL outlet.
2. **Backend Processing (FastAPI Server):** Connects to the LSL stream, extracts frequency band power features (Alpha, Beta, Theta), and maps them into cognitive state scores (Attention, Cognitive Load, Tension, Drowsiness).
3. **Adaptive Daemon (PyQt6):** An OS-level overlay daemon that tracks the user's focus and uses Windows UI Automation to enable "Dwell-clicking" and UI navigation based purely on the processed cognitive state.

## Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set your Gemini API key as an environment variable if you wish to use the AI wellness suggestions feature:
   ```bash
   set GEMINI_API_KEY="your_api_key_here"
   ```

## Running the Project

Because NAOS is a distributed pipeline, you need to start the components in the correct order.

1. **Start the NEST Simulator:**
   ```bash
   python nest_main_v9.py
   ```
   *Note: Click "Start Streaming" in the GUI to begin broadcasting LSL data.*

2. **Start the Backend Server:**
   Open a new terminal and run:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. **Start the OS Daemon (Optional):**
   Open a new terminal and run:
   ```bash
   python naos_daemon_v3.py
   ```

4. **View the Web Dashboard:**
   Open `dashboard.html` in your web browser to visualize the live cognitive states.
