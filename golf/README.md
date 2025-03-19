# Golf Score Analyzer

A modern web application that analyzes golf scorecards using OCR (Optical Character Recognition) technology. Upload a photo of your scorecard and get detailed statistics and visualizations of your golf performance.

## Features

- **OCR Processing**: Automatically extracts scores from scorecard images
- **Advanced Image Processing**:
  - Adaptive thresholding for better text recognition
  - Automatic image deskewing
  - Noise reduction
  - Enhanced OCR configuration for numerical data

- **Score Analysis**:
  - Hole-by-hole score tracking
  - Par comparison
  - Total score calculation
  - Statistics on birdies, pars, and bogeys
  - Average score per hole

- **Data Visualization**:
  - Interactive score vs par comparison chart
  - Strokes over/under par visualization
  - Color-coded performance indicators

- **Modern UI/UX**:
  - Responsive design
  - Drag-and-drop file upload
  - Bootstrap-based interface
  - Real-time feedback and error handling

- **Data Management**:
  - Automatic CSV export with timestamps
  - Data validation and error checking
  - Support for PNG, JPG, and JPEG formats

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd golf
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On macOS:
     ```bash
     brew install tesseract
     ```
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - On Windows:
     Download and install from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Start the Flask application:
   ```bash
   python golf.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```

3. Upload a scorecard image:
   - Use the drag-and-drop interface
   - Or click to select a file
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 16MB

4. View your results:
   - Score table with hole-by-hole breakdown
   - Performance statistics
   - Visual charts and graphs
   - Download your data as CSV

## Technical Details

- **Backend**: Python Flask
- **Image Processing**: OpenCV and Pillow
- **OCR Engine**: Tesseract
- **Data Analysis**: Pandas and NumPy
- **Visualization**: Matplotlib
- **Frontend**: Bootstrap 5, HTML5, JavaScript

## File Structure

```
golf/
├── golf.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── uploads/            # Directory for uploaded files
└── templates/          # HTML templates
    ├── upload.html    # File upload interface
    └── results.html   # Results display page
```

## Requirements

- Python 3.11 or higher
- Tesseract OCR
- See requirements.txt for Python package dependencies

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- File size limits
- OCR processing errors
- Data validation
- Server errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
