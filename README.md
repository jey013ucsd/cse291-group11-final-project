# cse291-group11-final-project


# Setup

## Setup Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## Download dataset
python get_dataset.py

## Preprocess dataset
python preprocess_dataset.py

- Images are converted to grayscale.  
- Each image is center-cropped to a square.  
- Cropped images are resized to 128x128 
- The processed image is saved as a **.png**, and its array + metadata as a **.pkl**.