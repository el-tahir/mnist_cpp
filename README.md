
# C++ MNIST Neural Network (From Scratch)


A simple neural network implementation that can learn MNIST in pure C++ without any machine learning libraries.

I realized I kinda have no idea how computers work, so here we are. Learned about SIMD, networking, serialization and ofcourse, some ML.

## Features

* **Core Math Engine:** Custom `Matrix` library implementing linear algebra operations.
* **SIMD Optimization:** AVX2 intrinsics (`_mm256`) for matrix multiplication, achieving a **6x speedup** over standard scalar loops.
* **Serialization:** Custom binary file format for saving/loading trained weights.
* **Client-Server Architecture:**
    * **Backend:** C++ TCP Server (Linux Sockets) handling inference.
    * **Frontend:** Python (Tkinter) GUI for drawing digits and visualizing predictions in real-time.
* **Zero Dependencies:** Uses only the C++ Standard Library and POSIX networking headers.

## Performance

Training on MNIST (60,000 images, 20 epochs):

Standard loops: 187 seconds
AVX optimized : 31 seconds


## Build & Usage

### Prerequisites
* **OS:** Linux, macOS (Intel or Apple Silicon), or Windows (WSL 2).
* **Compiler:** `g++` or `clang++` (must support C++17).
* **Hardware:** Matrix::dot uses AVX SIMD on x86/x64, otherwise it will fall back to normal scalar operations.

* **Tools:** `make`, Python 3 (for the GUI client).

### 1. Compile


```bash
make
```

### 2. Train the Model

Run the executable. If no `model.bin` is found, it will start training.

```bash
./nn
```


After training, the program starts a TCP server on port `8080`.

### 4. Start the Client

Open a new terminal and launch the Python GUI.

```bash
python3 client.py
```
Draw a digit and click "Predict" to send the raw pixels to the C++ backend.

## Project Structure

  * `Matrix.cpp`: Linear algebra.
  * `NeuralNetwork.cpp`: Forward/Backward propagation and gradient descent.
  * `main.cpp`: Training loop and TCP Server logic.
  * `utils.cpp`: MNIST binary file parsing and ASCII visualization.

## Future Work

  * Implement convolutions (im2col)
 
