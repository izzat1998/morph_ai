# ReportLab Installation Guide

## Issue Description

The application is encountering an error when trying to import the 'reportlab' module:

```
ModuleNotFoundError: No module named 'reportlab'
```

This error occurs because the 'reportlab' package, which is required for PDF generation functionality, is not installed in your Python environment.

## Solution

### Option 1: Using the Installation Script

1. Run the provided installation script:
   ```bash
   ./install_reportlab.sh
   ```

   This script will:
   - Activate your virtual environment (if it exists in the `.venv` directory)
   - Install the reportlab package with the correct version (4.0.9)

2. After installation, try running your application again.

### Option 2: Manual Installation

If you prefer to install the package manually:

1. Activate your virtual environment (if you're using one):
   ```bash
   source .venv/bin/activate
   ```

2. Install reportlab using pip:
   ```bash
   pip install reportlab==4.0.9
   ```

### Option 3: Install All Dependencies

To install all project dependencies:

1. Activate your virtual environment (if you're using one):
   ```bash
   source .venv/bin/activate
   ```

2. Install all dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Verification

After installation, you should be able to run the application without encountering the "No module named 'reportlab'" error.