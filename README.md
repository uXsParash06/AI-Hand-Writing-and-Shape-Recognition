# AI Hand Writing and Shape Recognition

An AI-based virtual handwriting application that uses
hand gestures and computer vision to write in the air
using a webcam.

## Project Overview

This application tracks hand movements using MediaPipe
and converts index-finger movements into digital strokes.

It also provides shape recognition and handwritten
text recognition.

## Features

- Real-time hand tracking
- 21 hand landmark detection
- Index finger-based virtual writing
- Closed fist to stop writing
- Open palm to clear the drawing
- Shape recognition
- Handwriting recognition using Tesseract OCR
- Live webcam display

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Tesseract OCR
- Pytesseract

## Project Structure

```text
AI-Hand-Writing-Recognition/
|
|-- AiHandPen.py
|-- hand_landmarker.task
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- screenshots/
```

## Requirements

- Python 3.13 (use a compatible MediaPipe release)
- Webcam
- Tesseract OCR
- Internet connection for initial package installation

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/uXsParash06/AI-Hand-Writing-and-Shape-Recognition.git
```

### 2. Open the project folder

```bash
cd AI-Hand-Writing-Recognition
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR separately on Windows.

Official project:
https://github.com/tesseract-ocr/tesseract

Update the executable path in AiHandPen.py if necessary.

### 5. Run the application

```bash
python AiHandPen.py
```

## Controls

| Gesture / Key         | Function         |
| --------------------- | ---------------- |
| Index finger extended | Start writing    |
| Closed fist           | Stop writing     |
| Open palm             | Clear drawing    |
| S                     | Recognize shape  |
| T                     | Recognize text   |
| C                     | Clear drawing    |
| Q                     | Quit application |

## Shape Recognition

The application identifies:

- Triangle
- Square
- Rectangle
- Circle
- Polygon

## Text Recognition

Tesseract OCR converts processed handwriting images
into machine-readable text.

## Future Improvements

- Improved continuous hand tracking
- Better handwriting recognition
- Support for multiple gestures
- Improved shape classification
- More accurate OCR

## Author

Your Name

## License

This project is available under the MIT License.
