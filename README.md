# FaceID: AI-Based Facial Recognition Attendance & Identity System

![FaceID Banner](https://via.placeholder.com/1200x300.png?text=FaceID+Facial+Recognition+System)

## Overview

FaceID is a professional, Python-based desktop application designed for real-time facial recognition and automated attendance tracking. Using your computer's webcam, the application can detect faces, register new individuals, and recognize them in real-time. All recognition events are logged locally, making it an ideal solution for classrooms, offices, or personal security.

## Features

- **Real-Time Detection & Recognition:** Uses dlib's state-of-the-art HOG face detector and a deep learning-based 128-d facial encoding model to recognize faces with high accuracy.
- **Automated Attendance Logging:** Records the exact date and time a person is recognized and exports logs to a CSV file (`attendance.csv`).
- **Interactive GUI:** A modern, dark-themed user interface built with CustomTkinter, featuring a live video feed, real-time bounding boxes, and an activity log.
- **Voice Greetings:** Integrated text-to-speech (TTS) welcomes registered users by name when they are recognized.
- **Persistent Storage:** Uses a local SQLite database to store face encodings, user data, and recognition statistics securely.

## Screenshots

*(Placeholders for future screenshots)*

| Main Dashboard | Registration Dialog |
| :---: | :---: |
| ![Dashboard](https://via.placeholder.com/400x300.png?text=Main+Dashboard) | ![Registration](https://via.placeholder.com/400x300.png?text=Registration+Dialog) |

## Technologies Used

- **Python 3.12+**
- **[OpenCV](https://opencv.org/):** Video capture, image processing, and format conversion.
- **[dlib](http://dlib.net/):** HOG face detection.
- **[face_recognition](https://github.com/ageitgey/face_recognition):** Generating 128-d face encodings and comparing faces.
- **[CustomTkinter](https://customtkinter.tomschimansky.com/):** Modern hardware-accelerated UI framework.
- **[pyttsx3](https://pyttsx3.readthedocs.io/):** Offline text-to-speech synthesis.
- **SQLite3:** Built-in lightweight database for persistent storage.

## Installation

### Prerequisites
- Python 3.12 or newer.
- A functional webcam.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/faceid.git
   cd faceid
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note for Windows Users:** Installing `dlib` via pip usually requires C++ Build Tools. If the standard installation fails, you can install a pre-compiled wheel for Python 3.12.

## Usage

1. Start the application by running the `run.py` script from the project root:
   ```bash
   python run.py
   ```
2. The application will launch and activate your default webcam.
3. If an unrecognized face is detected, click the **"Register New Face"** button. The system will prompt you for a name and capture a few frames to learn the face.
4. Once registered, the system will automatically recognize the person when they appear on camera, greet them via audio, and log their attendance in `data/attendance.csv`.

## Project Structure

```text
faceid/
├── run.py                 # Application entry point
├── src/                   # Core application source code
│   ├── main.py            # Main GUI initialization
│   ├── gui.py             # CustomTkinter interface & video loop
│   ├── camera.py          # Background threaded video capture
│   ├── face_detector.py   # dlib HOG face localization
│   ├── face_recognition_engine.py # Encoding and identification logic
│   ├── database.py        # SQLite database operations
│   ├── attendance.py      # CSV attendance logging
│   ├── utils.py           # TTS and image processing utilities
│   └── settings.py        # Configuration manager
├── data/                  # Local storage (ignored in version control)
│   ├── faces.db           # SQLite database
│   ├── settings.json      # User preferences
│   └── attendance.csv     # Exported attendance logs
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This documentation
```

## Future Improvements

- [ ] Support for multiple camera sources / IP cameras.
- [ ] Transition to CNN-based face detection for even higher accuracy on GPU-enabled machines.
- [ ] Web dashboard for viewing attendance records remotely.
- [ ] Export attendance logs to PDF format.

## Credits & Attribution

This project utilizes several fantastic open-source libraries:
- Face recognition functionality is powered by Adam Geitgey's [face_recognition](https://github.com/ageitgey/face_recognition) library, which wraps Davis King's excellent [dlib](http://dlib.net/) library.
- UI elements were built using [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) by Tom Schimansky.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Name:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **LinkedIn:** [Your Profile](https://linkedin.com/in/yourprofile)
