# micTRNGgui: A real-time visualization of TRNG

This project is a **True Random Number Generator (TRNG) real-time visualization tool** that generates random numbers using physical analog noise input from a microphone. It extracts micro thermal and acoustic chaos from audio data down to the Least Significant Bit (LSB) and processes it through a cryptographic hash function (SHA-256) to generate a final, true 64-bit random integer.

---

## Key Features

* **Real-time Audio Streaming:** Uses `QPainter` to render high-speed, real-time waveforms without frame drops.


* **2-Second Snapshot & Frequency Analysis (FFT):** Buffers data every 2 seconds to provide `Matplotlib`-based waveform and frequency spectrum (FFT) graphs.


* **Signal Statistics & Entropy Calculation:** Evaluates noise quality by calculating real-time entropy ($$H = -\sum P(x_i) \log_2 P(x_i)$$) along with the amplitude (Max/Min), Mean, Standard Deviation (Std Dev), and RMS values of the collected noise.


* **LSB Extraction & Visualization:** Extracts the Least Significant Bits (LSBs), which contain the strongest physical randomness, from the sampled data and visualizes them as binary data in a table.


* **Cryptographic Post-Processing (SHA-256):** Hashes the collected data using the SHA-256 algorithm to debias and compress it, ultimately outputting a 64-bit integer random number.


* **Modern Synthwave Theme:** Features an intuitive UI based on a cyber-dark theme built with PyQt6, alongside a dynamic TRNG pipeline flowchart.


* **Data Export:** Extracted random numbers, estimated entropy, and cryptographic hash digests can be saved locally as a `.txt` file along with their timestamps.



---

## TRNG Pipeline

The random number generation goes through the following 5-step sequence:

1. **Analog Microphone:** Collects ambient white noise and physical thermal noise through the microphone.


2. **Digitized Samples:** Digitizes the analog signal at a 44.1kHz sample rate into a raw data array.


3. **LSB Harvesting:** Collects the least significant bits (LSBs) containing micro-geometric chaos from the sampled data to form a raw bitstream.


4. **SHA-256 Extraction:** Applies cryptographic hashing to condense the entropy of the collected bitstream and maximize its cryptographic quality.


5. **True Random Integer:** Converts the derived hash digest to output a final 64-bit true random integer.



---

## Dependencies

To run this project, you need to install the following libraries:

* `PyQt6` (GUI framework)


* `numpy` (Numerical computations)


* `scipy` (FFT and frequency analysis)


* `matplotlib` (Snapshot and graph visualization)


* `sounddevice` (Microphone audio data streaming)



---

## Installation & Usage

1. **Clone the Repository**
```bash
git clone https://github.com/movenb3at/micTRNGgui.git
cd micTRNGgui

```


2. **Install Dependencies**
```bash
pip install PyQt6 numpy scipy matplotlib sounddevice

```


3. **Run the Application**
Ensure that a microphone device is properly connected to your system, then run the main script (e.g., `main.py`).


```bash
python main.py

```


4. **How to Use**
* Click the **[START]** button at the top to begin real-time waveform visualization and random number analysis.


* To halt the process, click the **[STOP]** button.


* Successfully generated random number data can be saved locally by clicking the **[SAVE RANDOM NUMBER]** button.


* You can monitor the real-time operational status and potential errors through the **PROCESS EXECUTION LOG** window at the bottom.





---

## ⚠️ Important Notes

* **Microphone Permissions and Physical Device Required:** This program utilizes physical input from a hardware sensor (microphone) as its absolute entropy source. Therefore, if a microphone is not connected or if access permissions are blocked, the collection process will not start, and an error log will force the operation to abort.