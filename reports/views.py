"""
Views for PDF report generation.
"""

import json
import logging
from typing import Dict, Any

from django.http import HttpResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.core.cache import cache

from cells.models import CellAnalysis, DetectedCell
from .pdf_generator import MorphometricPDFReport

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET"])
def generate_analysis_pdf(request, analysis_id: int):
    """
    Generate and download a comprehensive PDF report for the analysis.
    
    Args:
        request: HTTP request
        analysis_id: ID of the analysis to generate report for
        
    Returns:
        HttpResponse with PDF content or error response
    """
    try:
        # Get analysis
        analysis = get_object_or_404(CellAnalysis, id=analysis_id)
        
        # Check permissions (user should own the cell or be staff)
        if not (request.user == analysis.cell.user or request.user.is_staff):
            raise Http404("Analysis not found")
        
        # Check if analysis is completed
        if analysis.status != 'completed':
            return JsonResponse({
                'error': 'Analysis must be completed before generating report',
                'status': analysis.status
            }, status=400)
        
        # Get report configuration from request parameters
        config = {
            'include_methodology': request.GET.get('methodology', 'true').lower() == 'true',
            'include_individual_cells': request.GET.get('individual_cells', 'true').lower() == 'true',
            'include_charts': request.GET.get('charts', 'true').lower() == 'true',
            'include_quality_control': request.GET.get('quality_control', 'true').lower() == 'true',
            'max_cells_per_page': max(10, min(100, int(request.GET.get('max_cells_per_page', 30)))),
        }
        
        # Check cache for existing report
        cache_key = f"pdf_report_{analysis_id}_{hash(str(config))}"
        cached_pdf = cache.get(cache_key)
        
        if cached_pdf:
            logger.info(f"Serving cached PDF report for analysis {analysis_id}")
            response = HttpResponse(cached_pdf, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="morphometric_report_{analysis.cell.name}_{analysis_id}.pdf"'
            return response
        
        # Generate new report
        logger.info(f"Generating PDF report for analysis {analysis_id}")
        
        try:
            report_generator = MorphometricPDFReport(analysis, config)
            pdf_buffer = report_generator.generate_report()
            pdf_content = pdf_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error in PDF generation: {e}")
            # Return error response instead of trying to cache
            return JsonResponse({
                'error': 'PDF generation failed',
                'message': str(e),
                'details': 'Please check server logs for more information'
            }, status=500)
        
        # Cache the report for 1 hour
        cache.set(cache_key, pdf_content, 3600)
        
        # Create response
        response = HttpResponse(pdf_content, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="morphometric_report_{analysis.cell.name}_{analysis_id}.pdf"'
        response['Content-Length'] = len(pdf_content)
        
        logger.info(f"Successfully generated PDF report for analysis {analysis_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating PDF report for analysis {analysis_id}: {e}")
        return JsonResponse({
            'error': 'Failed to generate PDF report',
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def preview_analysis_pdf(request, analysis_id: int):
    """
    Generate a preview of the PDF report (first few pages).
    
    Args:
        request: HTTP request
        analysis_id: ID of the analysis
        
    Returns:
        JsonResponse with preview information
    """
    try:
        # Get analysis
        analysis = get_object_or_404(CellAnalysis, id=analysis_id)
        
        # Check permissions
        if not (request.user == analysis.cell.user or request.user.is_staff):
            raise Http404("Analysis not found")
        
        # Check if analysis is completed
        if analysis.status != 'completed':
            return JsonResponse({
                'error': 'Analysis must be completed before previewing report',
                'status': analysis.status
            }, status=400)
        
        # Get detected cells for preview statistics
        detected_cells = DetectedCell.objects.filter(analysis=analysis)
        
        preview_data = {
            'analysis_id': analysis.id,
            'cell_name': analysis.cell.name,
            'analysis_date': analysis.analysis_date.isoformat(),
            'status': analysis.get_status_display(),
            'total_cells': detected_cells.count(),
            'processing_time': analysis.processing_time,
            'scale_calibrated': analysis.cell.scale_set,
            'has_visualizations': {
                'segmentation': bool(analysis.segmentation_image),
                'flow_analysis': bool(analysis.flow_analysis_image),
                'style_quality': bool(analysis.style_quality_image),
                'edge_boundary': bool(analysis.edge_boundary_image),
            },
            'estimated_pages': _estimate_report_pages(analysis, detected_cells.count()),
            'file_size_estimate': _estimate_file_size(analysis, detected_cells.count()),
        }
        
        return JsonResponse(preview_data)
        
    except Exception as e:
        logger.error(f"Error generating PDF preview for analysis {analysis_id}: {e}")
        return JsonResponse({
            'error': 'Failed to generate PDF preview',
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET", "POST"])
def configure_analysis_pdf(request, analysis_id: int):
    """
    Configure PDF report generation options.
    
    Args:
        request: HTTP request
        analysis_id: ID of the analysis
        
    Returns:
        Rendered template or JsonResponse with configuration
    """
    try:
        # Get analysis
        analysis = get_object_or_404(CellAnalysis, id=analysis_id)
        
        # Check permissions
        if not (request.user == analysis.cell.user or request.user.is_staff):
            raise Http404("Analysis not found")
        
        if request.method == 'GET':
            # Return configuration form
            detected_cells = DetectedCell.objects.filter(analysis=analysis)
            
            context = {
                'analysis': analysis,
                'total_cells': detected_cells.count(),
                'has_visualizations': {
                    'segmentation': bool(analysis.segmentation_image),
                    'flow_analysis': bool(analysis.flow_analysis_image),
                    'style_quality': bool(analysis.style_quality_image),
                    'edge_boundary': bool(analysis.edge_boundary_image),
                },
                'default_config': {
                    'include_methodology': True,
                    'include_individual_cells': True,
                    'include_charts': True,
                    'include_quality_control': True,
                    'max_cells_per_page': 30,
                }
            }
            
            return render(request, 'reports/configure_pdf.html', context)
        
        elif request.method == 'POST':
            # Process configuration and estimate report properties
            try:
                config = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
            
            # Ensure max_cells_per_page is an integer
            if 'max_cells_per_page' in config:
                try:
                    config['max_cells_per_page'] = int(config['max_cells_per_page'])
                except (ValueError, TypeError):
                    config['max_cells_per_page'] = 30  # Default fallback
            
            detected_cells = DetectedCell.objects.filter(analysis=analysis)
            
            # Estimate report properties with given configuration
            estimated_pages = _estimate_report_pages(analysis, detected_cells.count(), config)
            estimated_size = _estimate_file_size(analysis, detected_cells.count(), config)
            estimated_time = _estimate_generation_time(analysis, detected_cells.count(), config)
            
            response_data = {
                'estimated_pages': estimated_pages,
                'estimated_file_size': estimated_size,
                'estimated_generation_time': estimated_time,
                'configuration_valid': True,
            }
            
            return JsonResponse(response_data)
            
    except Exception as e:
        logger.error(f"Error configuring PDF for analysis {analysis_id}: {e}")
        return JsonResponse({
            'error': 'Failed to configure PDF report',
            'message': str(e)
        }, status=500)


def _estimate_report_pages(analysis: CellAnalysis, cell_count: int, config: Dict[str, Any] = None) -> int:
    """
    Estimate the number of pages in the generated report.
    
    Args:
        analysis: CellAnalysis instance
        cell_count: Number of detected cells
        config: Report configuration
        
    Returns:
        Estimated number of pages
    """
    if config is None:
        config = {
            'include_methodology': True,
            'include_individual_cells': True,
            'include_charts': True,
            'include_quality_control': True,
            'max_cells_per_page': 30,
        }
    
    pages = 0
    
    # Cover page + TOC
    pages += 2
    
    # Executive summary
    pages += 1
    
    # Methodology
    if config.get('include_methodology', True):
        pages += 1
    
    # Image processing pipeline
    visualization_count = sum([
        bool(analysis.segmentation_image),
        bool(analysis.flow_analysis_image),
        bool(analysis.style_quality_image),
        bool(analysis.edge_boundary_image),
    ])
    pages += max(1, (visualization_count + 1) // 2)  # 2 images per page approximately
    
    # Statistical analysis
    if config.get('include_charts', True) and cell_count > 0:
        pages += 3  # Distributions, correlations, box plots
    
    # Individual cell results
    if config.get('include_individual_cells', True) and cell_count > 0:
        max_cells_per_page = int(config.get('max_cells_per_page', 30))
        max_cells_per_page = max(10, min(100, max_cells_per_page))  # Ensure valid range
        pages += max(1, (cell_count + max_cells_per_page - 1) // max_cells_per_page)
    
    # Quality control
    if config.get('include_quality_control', True):
        pages += 1
    
    # Technical appendix
    pages += 1
    
    return pages


def _estimate_file_size(analysis: CellAnalysis, cell_count: int, config: Dict[str, Any] = None) -> str:
    """
    Estimate the file size of the generated report.
    
    Args:
        analysis: CellAnalysis instance
        cell_count: Number of detected cells
        config: Report configuration
        
    Returns:
        Estimated file size as string (e.g., "2.5 MB")
    """
    if config is None:
        config = {
            'include_methodology': True,
            'include_individual_cells': True,
            'include_charts': True,
            'include_quality_control': True,
            'max_cells_per_page': 30,
        }
    
    # Base size for text content (KB)
    base_size = 100
    
    # Add size for visualizations (each ~300KB at 300dpi)
    visualization_count = sum([
        bool(analysis.segmentation_image),
        bool(analysis.flow_analysis_image),
        bool(analysis.style_quality_image),
        bool(analysis.edge_boundary_image),
    ])
    visualization_size = visualization_count * 300
    
    # Add size for charts (~200KB each)
    if config.get('include_charts', True) and cell_count > 0:
        chart_size = 6 * 200  # 6 charts approximately
    else:
        chart_size = 0
    
    # Add size for individual cell data (minimal)
    if config.get('include_individual_cells', True):
        data_size = max(50, cell_count * 0.1)  # Very rough estimate
    else:
        data_size = 0
    
    # Total size in KB
    total_size_kb = base_size + visualization_size + chart_size + data_size
    
    # Convert to appropriate units
    if total_size_kb < 1024:
        return f"{total_size_kb:.0f} KB"
    else:
        total_size_mb = total_size_kb / 1024
        return f"{total_size_mb:.1f} MB"


def _estimate_generation_time(analysis: CellAnalysis, cell_count: int, config: Dict[str, Any] = None) -> str:
    """
    Estimate the time required to generate the report.
    
    Args:
        analysis: CellAnalysis instance
        cell_count: Number of detected cells
        config: Report configuration
        
    Returns:
        Estimated generation time as string (e.g., "30 seconds")
    """
    if config is None:
        config = {
            'include_methodology': True,
            'include_individual_cells': True,
            'include_charts': True,
            'include_quality_control': True,
            'max_cells_per_page': 30,
        }
    
    # Base time for report structure (seconds)
    base_time = 5
    
    # Add time for chart generation
    if config.get('include_charts', True) and cell_count > 0:
        chart_time = 10  # ~10 seconds for all charts
    else:
        chart_time = 0
    
    # Add time for large datasets
    if config.get('include_individual_cells', True) and cell_count > 1000:
        data_time = (cell_count // 1000) * 5  # Extra time for large datasets
    else:
        data_time = 0
    
    # Add time for image processing
    visualization_count = sum([
        bool(analysis.segmentation_image),
        bool(analysis.flow_analysis_image),
        bool(analysis.style_quality_image),
        bool(analysis.edge_boundary_image),
    ])
    image_time = visualization_count * 2  # 2 seconds per image
    
    total_time = base_time + chart_time + data_time + image_time
    
    if total_time < 60:
        return f"{total_time:.0f} seconds"
    else:
        minutes = total_time // 60
        seconds = total_time % 60
        if seconds > 0:
            return f"{minutes:.0f}m {seconds:.0f}s"
        else:
            return f"{minutes:.0f} minutes"