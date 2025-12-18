# Object Tracking Baseline (BIL539)

This repository contains the baseline implementation for the BIL539 term project.  
The baseline uses a template-matching tracker based on `cv2.TM_CCOEFF_NORMED`.

## Project Structure
```
.
├── baseline.py
├── images/
│   └── MVI_20011/
│       └── *.jpg
├── annotations/
│   └── MVI_20011.xml
└── requirements.txt
```

## Environment
- Python 3.10
- OpenCV 4.9
- NumPy 1.26
- ElementTree (built-in)

Install dependencies:
```
pip install -r requirements.txt
```

## Run
```
python baseline.py
```

## Description
- Extracts template from the first valid ground-truth frame  
- Performs full-frame normalized cross-correlation  
- Outputs IoU, CLE, and FPS metrics  
- No motion model or template update  
