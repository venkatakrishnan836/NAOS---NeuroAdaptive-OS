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
