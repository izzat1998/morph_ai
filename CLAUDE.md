# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morph AI is a Django-based web application for morphometric analysis of cell images using Cellpose segmentation. The application allows users to upload cell images, run automated analysis, and extract detailed morphometric features.

## Key Architecture

### Core Applications
- **accounts**: Custom user authentication with email-based login (AUTH_USER_MODEL = 'accounts.User')
- **cells**: Main application handling cell image uploads, analysis processing, and morphometric feature extraction

### Analysis Pipeline
The morphometric analysis pipeline in `cells/analysis.py` follows this workflow:
1. Image preprocessing using PIL and cellpose.io
2. Cellpose segmentation with configurable models (cyto, nuclei, cyto2, custom)
3. Feature extraction using scikit-image regionprops
4. Statistical analysis and visualization generation

### Database Schema
- **Cell**: Stores uploaded images with metadata (dimensions, file format, analysis tracking)
- **CellAnalysis**: Analysis parameters and results (cellpose model, diameter, thresholds, processing status)
- **DetectedCell**: Individual cell measurements (area, perimeter, shape descriptors, ellipse fitting)

## Development Commands

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database Operations
```bash
# Create and apply migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### Running the Application
```bash
# Development server
python manage.py runserver

# Collect static files (for production)
python manage.py collectstatic
```

### Testing
```bash
# Run all tests
python manage.py test

# Run tests for specific app
python manage.py test accounts
python manage.py test cells

# Run GPU acceleration tests
python manage.py test cells.tests.test_gpu_acceleration
```

### GPU Benchmarking
```bash
# Run GPU performance benchmark
python manage.py gpu_benchmark

# Benchmark specific operations
python manage.py gpu_benchmark --operation morphometrics
python manage.py gpu_benchmark --operation preprocessing

# Run comprehensive benchmark suite
python manage.py gpu_benchmark --full-suite --iterations 10
```

## Key Dependencies

### Core Framework
- Django 5.2.3 with PostgreSQL backend (psycopg2-binary)
- Custom user model with email authentication
- Bootstrap 5 frontend with django-crispy-forms

### Scientific Computing
- **cellpose[gui]**: Deep learning-based cell segmentation
- **scikit-image**: Image processing and morphometric feature extraction
- **numpy/scipy**: Numerical computations
- **matplotlib**: Visualization generation
- **pillow**: Image handling

### Production
- gunicorn for WSGI serving
- whitenoise for static file serving
- Security middleware configured for production deployment

## Analysis Features

The morphometric analysis extracts these features for each detected cell:
- Basic measurements: area, perimeter
- Shape descriptors: circularity, eccentricity, solidity, extent
- Ellipse fitting: major/minor axis lengths, aspect ratio
- Spatial data: centroid coordinates, bounding box

## Environment Configuration

Required environment variables (see .env.example):
- Database: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- Django: SECRET_KEY, DEBUG, ALLOWED_HOSTS
- Optional: Email settings for notifications

### GPU Acceleration Settings
- **CELLPOSE_USE_GPU**: Enable GPU for Cellpose segmentation ('auto', 'true', 'false')
- **ENABLE_GPU_PREPROCESSING**: Enable GPU-accelerated image preprocessing ('auto', 'true', 'false')
- **GPU_MEMORY_FRACTION**: Fraction of GPU memory to use (default: 0.8)
- **GPU_BATCH_SIZE**: Batch size for GPU operations (default: 4)

## Internationalization

The application supports multiple languages (English, Russian, Uzbek) with locale files in the `locale/` directory. Use Django's translation utilities for adding new translatable strings.

## Media Handling

- Cell images uploaded to `media/cells/`
- Analysis visualizations saved to `media/analyses/segmentation/`
- Automatic metadata extraction (dimensions, file size, format) on upload