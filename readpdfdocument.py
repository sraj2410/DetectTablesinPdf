import cv2
import numpy as np
from pdf2image import convert_from_path

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image
    inverted = cv2.bitwise_not(thresh)

    return inverted

def detect_tables_and_charts(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        preprocessed_image = preprocess_image(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        # Find contours in the preprocessed image
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        # Check if there are enough valid contours (indicative of tables or charts)
        if len(valid_contours) >= 3:  # You may need to adjust this threshold based on your PDFs
            return True

    return False

# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_path = "/Users/rajeev/Desktop/SouthwestConfirmation.pdf"

if detect_tables_and_charts(pdf_path):
    print("The PDF contains tables and charts.")
else:
    print("No tables or charts found in the PDF.")
