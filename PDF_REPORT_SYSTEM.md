# Comprehensive PDF Report Generation System

## Overview

This system provides professional, publication-ready PDF reports for morphometric analysis results. The reports include complete visualizations, statistical analysis, individual cell data, and detailed methodology sections.

## Features

### 📋 Report Sections
- **Cover Page**: Analysis metadata, cell information, and summary
- **Executive Summary**: Key findings and quality metrics
- **Methodology**: Detailed processing parameters and pipeline description
- **Image Processing Pipeline**: All available visualizations (up to 18 panels)
- **Statistical Analysis**: Histograms, correlations, box plots, and summary tables
- **Individual Cell Results**: Complete measurements table with pagination
- **Quality Control**: Processing validation and filtering information
- **Technical Appendix**: Complete parameter documentation

### 📊 Statistical Charts
- Distribution histograms with statistical overlays
- Correlation matrix heatmaps
- Box plot comparisons
- Scatter plots with regression lines
- Quality metrics radar charts
- Professional summary statistics tables

### 🎛️ Customization Options
- Section inclusion/exclusion
- Data presentation options
- Report quality settings (72-300 DPI)
- Cells per page configuration
- Progress tracking and estimation

## Installation

### 1. Dependencies
The system automatically adds these dependencies to `requirements.txt`:
```
reportlab==4.0.9
seaborn==0.13.2
pandas==2.2.3
```

### 2. Django Configuration
The reports app is automatically added to:
- `INSTALLED_APPS` in settings.py
- URL configuration in main urls.py

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Quick PDF Generation
1. Navigate to any completed analysis detail page
2. Click "Export" → "Quick PDF Report"
3. PDF downloads automatically

### Configured PDF Generation
1. Navigate to analysis detail page
2. Click "Export" → "Configure PDF Report" 
3. Select desired sections and options
4. Preview configuration
5. Generate customized report

### Direct URL Access
```
/reports/analysis/<analysis_id>/pdf/                    # Quick generate
/reports/analysis/<analysis_id>/pdf/config/             # Configuration page
/reports/analysis/<analysis_id>/pdf/preview/            # Preview info (JSON)
```

## API Endpoints

### Generate PDF Report
```
GET /reports/analysis/<id>/pdf/
Parameters:
- methodology: true/false (include methodology section)
- individual_cells: true/false (include cell data table)
- charts: true/false (include statistical charts)
- quality_control: true/false (include QC section)
- max_cells_per_page: integer (cells per page in tables)
```

### Get Report Configuration
```
GET /reports/analysis/<id>/pdf/config/
Returns: Configuration form template

POST /reports/analysis/<id>/pdf/config/
Body: JSON configuration object
Returns: Estimated report properties
```

### Preview Report Properties
```
GET /reports/analysis/<id>/pdf/preview/
Returns: JSON with estimated pages, file size, generation time
```

## Report Structure

### Standard Report Includes:
1. **Cover Page** (1 page)
   - Analysis title and metadata
   - Cell image preview
   - Analysis summary statistics

2. **Table of Contents** (1 page)
   - Hierarchical navigation

3. **Executive Summary** (1 page)
   - Key findings
   - Quality assessment
   - Processing summary

4. **Methodology** (1 page, optional)
   - Cellpose parameters
   - Processing pipeline
   - Preprocessing steps

5. **Image Processing Pipeline** (2-4 pages)
   - Core 6-panel visualization
   - Flow analysis (4 panels)
   - Style & quality analysis (4 panels)
   - Edge & boundary analysis (4 panels)

6. **Statistical Analysis** (3 pages, optional)
   - Distribution histograms
   - Correlation matrices
   - Box plots and summary tables

7. **Individual Cell Results** (variable pages, optional)
   - Complete measurements table
   - Configurable pagination (10-100 cells/page)

8. **Quality Control** (1 page, optional)
   - Processing validation
   - Filtering information
   - Quality metrics

9. **Technical Appendix** (1 page)
   - Complete parameter listing
   - Software information
   - Generation metadata

## Performance

### Caching
- Reports cached for 1 hour using configuration hash
- Reduces regeneration time for identical requests

### Generation Times
- Small datasets (<100 cells): 15-30 seconds
- Medium datasets (100-1000 cells): 30-60 seconds  
- Large datasets (>1000 cells): 1-3 minutes

### File Sizes
- Typical report: 2-5 MB
- With all visualizations: 5-8 MB
- Large datasets: up to 15 MB

## Customization

### Chart Styling
Edit `reports/charts.py` to modify:
- Color schemes
- Chart types
- Statistical overlays
- Professional styling

### Report Layout
Edit `reports/pdf_generator.py` to modify:
- Page layouts
- Section organization
- Header/footer content
- Typography

### UI Integration
Edit templates to add PDF export buttons:
- `templates/cells/analysis_detail.html`
- `templates/reports/configure_pdf.html`

## Error Handling

### Graceful Fallbacks
- Missing visualizations: Automatically skipped
- Chart generation errors: Error placeholders inserted
- Large datasets: Automatic pagination
- Memory issues: Optimized image compression

### Logging
All operations logged to Django logger `reports`

## Security

### Access Control
- User must own the analysis or be staff
- Django login required
- CSRF protection on configuration

### File Safety
- PDF generation isolated
- No file system access beyond media
- Temporary chart images automatically cleaned

## Development

### Adding New Chart Types
1. Add method to `MorphometricChartGenerator` class
2. Call from `MorphometricPDFReport._add_statistical_analysis()`
3. Handle errors gracefully

### Adding Report Sections
1. Create method in `MorphometricPDFReport` class
2. Add to `generate_report()` method
3. Update configuration options

### Extending Configuration
1. Add options to configuration form
2. Update view parameter handling
3. Pass to report generator

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Memory errors**: Reduce image quality or limit cells per page
3. **Generation timeout**: Increase request timeout for large datasets
4. **Missing visualizations**: Check analysis completion status

### Debug Mode
Enable Django debug mode to see detailed error messages in PDF generation.