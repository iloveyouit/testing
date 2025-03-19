from flask import Flask, request, render_template, flash, send_from_directory, redirect, url_for
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'golf_score_analyzer_secret'  # Required for flash messages

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Enhanced image preprocessing for better OCR results"""
    try:
        logger.debug(f"Reading image from {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image or file not found.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted image to grayscale")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        logger.debug("Applied Gaussian blur")
        
        # Apply adaptive thresholding with smaller block size
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 11
        )
        logger.debug("Applied adaptive thresholding")
        
        # Dilate to connect components
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        logger.debug("Applied dilation")
        
        # Save debug image
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_preprocessed.png')
        cv2.imwrite(debug_path, dilated)
        logger.debug(f"Saved debug image to {debug_path}")
        
        return dilated
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def extract_text_from_image(image_path):
    """Extract text with enhanced OCR configuration"""
    try:
        logger.debug("Starting OCR process")
        processed_image = preprocess_image(image_path)
        pil_image = Image.fromarray(processed_image)
        
        # Try different PSM modes and configurations
        configs = [
            '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 --dpi 300',  # Uniform block
            '--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789 --dpi 300',  # Column of text
            '--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789 --dpi 300',  # Single line
            '--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789 --dpi 300',  # Auto
        ]
        
        best_text = ""
        max_numbers = 0
        
        for config in configs:
            current_text = pytesseract.image_to_string(pil_image, config=config)
            logger.debug(f"Config {config} extracted: {current_text}")
            
            # Count valid numbers in the extracted text
            numbers = [n for n in current_text.split() if n.isdigit() and len(n) <= 2]
            if len(numbers) > max_numbers:
                max_numbers = len(numbers)
                best_text = current_text
        
        if not best_text.strip():
            raise ValueError("No text could be extracted with any configuration")
        
        logger.debug(f"Best extracted text: {best_text}")
        return best_text
    except Exception as e:
        logger.error(f"Error in extract_text_from_image: {str(e)}")
        raise

def parse_golf_score(text):
    """Parse golf score with enhanced validation and error handling"""
    try:
        logger.debug(f"Parsing text: {text}")
        # Get all valid numbers (1 or 2 digits)
        numbers = [num for num in text.replace('\n', ' ').split() 
                  if num.strip().isdigit() and len(num.strip()) <= 2]
        logger.debug(f"Found numbers: {numbers}")
        
        if len(numbers) < 2:
            raise ValueError("Not enough valid numbers found in the image")
        
        score_data = []
        current_hole = 1
        i = 0
        
        while i < len(numbers) - 1 and current_hole <= 18:
            try:
                # Try to find a valid par and score combination
                for j in range(i, min(i + 3, len(numbers))):
                    par = int(numbers[j])
                    if 3 <= par <= 5:  # Valid par value found
                        # Look for score in the next few numbers
                        for k in range(j + 1, min(j + 3, len(numbers))):
                            score = int(numbers[k])
                            if 1 <= score <= 12:  # Valid score found
                                score_data.append([current_hole, par, score])
                                logger.debug(f"Valid score entry: Hole {current_hole}, Par {par}, Score {score}")
                                current_hole += 1
                                i = k + 1
                                break
                        break
                else:
                    i += 1
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing numbers at position {i}: {str(e)}")
                i += 1
        
        if not score_data:
            raise ValueError("No valid golf score data found in the extracted text")
        
        df = pd.DataFrame(score_data, columns=['Hole', 'Par', 'Score'])
        logger.debug(f"Parsed data:\n{df}")
        return df
    except Exception as e:
        logger.error(f"Error in parse_golf_score: {str(e)}")
        raise

def calculate_statistics(df):
    """Calculate golf performance statistics"""
    stats = {}
    try:
        if not df.empty and 'Score' in df.columns and 'Par' in df.columns:
            df['Over_Under'] = df['Score'] - df['Par']
            stats['total_par'] = int(df['Par'].sum())
            stats['total_score'] = int(df['Score'].sum())
            stats['over_under'] = int(df['Over_Under'].sum())
            stats['average_score'] = round(df['Score'].mean(), 1)
            stats['pars'] = int(df[df['Over_Under'] == 0].shape[0])
            stats['birdies'] = int(df[df['Over_Under'] < 0].shape[0])
            stats['bogeys'] = int(df[df['Over_Under'] > 0].shape[0])
            logger.debug(f"Calculated statistics: {stats}")
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
    return stats

def plot_performance(df):
    """Enhanced performance visualization"""
    if df.empty:
        return None
    
    try:
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Score vs Par plot
        ax1.plot(df['Hole'], df['Par'], marker='o', linestyle='-', label='Par', color='green')
        ax1.plot(df['Hole'], df['Score'], marker='s', linestyle='-', label='Score', color='blue')
        ax1.set_xlabel('Hole')
        ax1.set_ylabel('Score')
        ax1.set_title('Score vs Par by Hole')
        ax1.legend()
        ax1.grid(True)
        
        # Over/Under par plot
        over_under = df['Score'] - df['Par']
        colors = ['green' if x < 0 else 'red' if x > 0 else 'blue' for x in over_under]
        ax2.bar(df['Hole'], over_under, color=colors)
        ax2.set_xlabel('Hole')
        ax2.set_ylabel('Strokes Over/Under Par')
        ax2.set_title('Strokes Over/Under Par by Hole')
        ax2.grid(True)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error in plot_performance: {str(e)}")
        return None

def save_to_csv(df, filename='golf_scores.csv'):
    """Save scores with timestamp"""
    try:
        if not df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'golf_scores_{timestamp}.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            df.to_csv(filepath, index=False)
            logger.debug(f"Saved scores to {filepath}")
            return filename
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file uploaded', 'error')
                return render_template('upload.html')
            
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return render_template('upload.html')
            
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload a PNG or JPEG image.', 'error')
                return render_template('upload.html')
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.debug(f"Saved uploaded file to {file_path}")
            
            extracted_text = extract_text_from_image(file_path)
            if not extracted_text.strip():
                flash('No text could be extracted from the image. Please ensure the image is clear and contains visible numbers.', 'error')
                return render_template('upload.html')
            
            df = parse_golf_score(extracted_text)
            if df.empty:
                flash('No valid golf score data could be extracted from the image.', 'error')
                return render_template('upload.html')
            
            stats = calculate_statistics(df)
            plot_data = plot_performance(df)
            csv_filename = save_to_csv(df)
            
            logger.debug(f"Generated statistics: {stats}")
            logger.debug(f"Saved CSV as: {csv_filename}")
            
            return render_template(
                'results.html',
                table=df.to_html(classes='table table-striped table-hover'),
                plot_url=plot_data,
                stats=stats,
                csv_filename=csv_filename
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return render_template('upload.html')
    
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_csv(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        flash('Error downloading file', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
