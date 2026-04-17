# ESP32-S3 Multimodal TinyML: Dual AI Models on FreeRTOS

[![PlatformIO](https://img.shields.io/badge/Built_with-PlatformIO-orange?logo=platformio)](https://platformio.org/)
[![Edge Impulse](https://img.shields.io/badge/AI_Powered_by-Edge_Impulse-blue?logo=edge-impulse)](https://edgeimpulse.com/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3-green)](https://www.espressif.com/)

## 📌 Overview
This project demonstrates a **Multimodal TinyML (Machine Learning on Edge Devices)** system using the ESP32-S3 microcontroller. It concurrently runs two independent Neural Network models—Voice Recognition (Keyword Spotting) and Gesture Recognition—by leveraging the dual-core architecture of the ESP32-S3 and FreeRTOS.

### 🚀 Key Features
* **Dual AI Processing:** Runs two Edge Impulse models simultaneously without blocking.
* **Voice Command Recognition (Keyword Spotting):** Uses an INMP441 I2S microphone to detect specific keywords (e.g., "Turn On", "Turn Off").
* **Motion Gesture Recognition:** Uses an MPU6050 accelerometer/gyroscope over I2C to classify hand/device movements.
* **FreeRTOS Multitasking:** Optimized task allocation utilizing both Core 0 and Core 1 to prevent audio sampling dropouts and ensure real-time responsiveness.
* **C++ Namespace Isolation:** Implements a custom Python script solution to resolve C++ "Multiple Definition" linker errors commonly faced when integrating multiple Edge Impulse SDKs in a single project.

---

## 🛠️ Hardware Requirements & Wiring

* **Microcontroller:** ESP32-S3 (YoloUNO Board)
* **Microphone:** INMP441 (Omnidirectional I2S)
* **IMU Sensor:** MPU6050 (6-axis Accelerometer & Gyroscope)
* **Actuators:** Active Low Buzzer, Standard LED

### Pin Configuration (`global.h`)
| Component | Pin Function | ESP32-S3 GPIO |
| :--- | :--- | :--- |
| **INMP441 (Mic)** | I2S WS (L/R Clock) | `GPIO 1` |
| | I2S SCK (BCLK) | `GPIO 2` |
| | I2S SD (Data Out) | `GPIO 3` |
| **MPU6050 (IMU)** | I2C SDA | `GPIO 11` |
| | I2C SCL | `GPIO 12` |
| **Buzzer** | Signal (Active Low) | `GPIO 4` |
| **LED** | Signal | `GPIO 6` |

---

## 🧠 System Architecture & RTOS Design

To ensure the heavy continuous audio sampling does not interfere with IMU readings, tasks are strictly pinned to specific CPU cores using FreeRTOS:

* **Core 0 (Audio & Voice AI):**
    * `voice_task`: Handles I2S DMA buffering (16kHz, 16-bit). Allocates 32KB of RAM for the 1-second audio frame and runs the voice inference engine. Assigned higher priority due to strict timing constraints.
* **Core 1 (Sensors & System):**
    * `gesture_task`: Communicates with MPU6050 via I2C (50Hz sampling rate). Buffers data and runs the gesture inference engine.
    * `ledBlinky`: Background task for system status indication.

---

## 🚧 Overcoming "Multiple Definitions" Linker Error

**The Challenge:**
Edge Impulse exports C++ Arduino libraries with identical underlying SDK filenames and global function names (`run_classifier`, `ei_printf`, etc.). Including two unmodified models in one PlatformIO project causes fatal Linker errors (Multiple Definitions).

**The Solution:**
This project utilizes a custom Python isolation script (`rename_v3.py`) applied to the Voice Model library. 
1. The Gesture model is kept as the standard `ei_` namespace.
2. The script deeply traverses the Voice model's source code, automatically rewriting variables, macros, folders, and function definitions from `ei_` to `voice_ei_`.
3. This creates a completely isolated SDK instance, allowing both brains to safely compile and link into the final `firmware.elf`.

---

## 💻 Building and Flashing

### Prerequisites
1.  **VS Code** with the **PlatformIO** extension installed.
2.  Generated C++ Arduino libraries from your Edge Impulse projects.

### Setup Instructions
1.  Clone this repository.
2.  Extract your primary model (e.g., Gesture) into the `lib/` directory normally.
3.  Extract your secondary model (e.g., Voice) into a separate folder, run the namespace isolation Python script on it, and then move it to the `lib/` directory.
4.  Ensure `platformio.ini` has `lib_ldf_mode` configured correctly (avoid `deep+` if it breaks `Wire.h` dependencies; use explicit `lib_deps = Wire`).
5.  Click the **Build (✓)** button in PlatformIO.
6.  Connect your ESP32-S3 and click **Upload (→)**.

### Testing
Open the Serial Monitor at `115200` baud rate. You should see successful initialization of both the I2S microphone and the MPU6050, followed by real-time inference confidence scores printing to the console.

---
*Developed as an Embedded Systems & IoT Engineering Project.*