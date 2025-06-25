from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
import csv
import json
from .forms import CellUploadForm, CellAnalysisForm, ScaleCalibrationForm
from .models import Cell, CellAnalysis, DetectedCell
from .analysis import run_cell_analysis, get_analysis_summary, get_image_quality_summary
from .services import CellUploadService, CellAnalysisService, CellExportService, CellScaleService
from django.utils.translation import gettext as _

@login_required
def upload_cell(request):
    if request.method == 'POST':
        form = CellUploadForm(request.POST, request.FILES)
        if form.is_valid():
            cell = CellUploadService.create_cell(form, request.user)
            messages.success(request, f'Изображение клетки "{cell.name}" успешно загружено!')
            return redirect('cells:upload')
    else:
        form = CellUploadForm()
    
    return render(request, 'cells/upload.html', {'form': form})


@login_required
def cell_list(request):
    cells = Cell.objects.filter(user=request.user)
    return render(request, 'cells/list.html', {'cells': cells})


@login_required
def cell_detail(request, cell_id):
    cell = get_object_or_404(Cell, id=cell_id, user=request.user)
    analyses = cell.analyses.all().order_by('-analysis_date')
    
    context = {
        'cell': cell,
        'analyses': analyses,
        'latest_analysis': cell.latest_analysis,
    }
    return render(request, 'cells/detail.html', context)


@login_required
def analyze_cell(request, cell_id):
    cell = get_object_or_404(Cell, id=cell_id, user=request.user)
    
    if request.method == 'POST':
        form = CellAnalysisForm(request.POST)
        if form.is_valid():
            analysis = CellAnalysisService.create_analysis(form, cell, request)
            
            # Run analysis in the background (for now, synchronously)
            # In production, this should be done with Celery or similar
            success = CellAnalysisService.run_analysis(analysis.id)
            
            if success:
                messages.success(request, 'Анализ завершен успешно!')
                return redirect('cells:analysis_detail', analysis_id=analysis.id)
            else:
                messages.error(request, 'Ошибка анализа. Пожалуйста, проверьте детали ошибки.')
                return redirect('cells:analysis_detail', analysis_id=analysis.id)
    else:
        form = CellAnalysisForm()
    
    context = {
        'cell': cell,
        'form': form,
    }
    return render(request, 'cells/analyze.html', context)


@login_required
def analysis_detail(request, analysis_id):
    analysis = get_object_or_404(CellAnalysis, id=analysis_id, cell__user=request.user)
    context = CellAnalysisService.get_analysis_context(analysis, request)
    return render(request, 'cells/analysis_detail.html', context)


@login_required
def analysis_list(request):
    context = CellAnalysisService.get_analysis_list_context(request.user, request)
    return render(request, 'cells/analysis_list.html', context)


@login_required
def analysis_status(request, analysis_id):
    """AJAX endpoint to check analysis status"""
    analysis = get_object_or_404(CellAnalysis, id=analysis_id, cell__user=request.user)
    
    data = {
        'status': analysis.status,
        'num_cells_detected': analysis.num_cells_detected,
        'processing_time': analysis.processing_time,
        'error_message': analysis.error_message,
    }
    
    return JsonResponse(data)


@login_required
def export_analysis_csv(request, analysis_id):
    """Export analysis results as CSV"""
    analysis = get_object_or_404(CellAnalysis, id=analysis_id, cell__user=request.user)
    
    if analysis.status != 'completed':
        messages.error(request, _('Analysis must be completed before export.'))
        return redirect('cells:analysis_detail', analysis_id=analysis_id)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="analysis_{analysis_id}_cells.csv"'
    
    writer = csv.writer(response)
    
    # Write header - include physical measurements if scale is available
    header = [
        _('Cell ID'), _('Area (px²)'), _('Perimeter (px)'), _('Circularity'), _('Eccentricity'),
        _('Solidity'), _('Extent'), _('Major Axis (px)'), _('Minor Axis (px)'), _('Aspect Ratio'),
        _('Centroid X'), _('Centroid Y'), _('Bbox X'), _('Bbox Y'), _('Bbox Width'), _('Bbox Height')
    ]
    
    # Add GLCM texture features to header
    header.extend([
        _('GLCM Contrast'), _('GLCM Correlation'), _('GLCM Energy'), _('GLCM Homogeneity'),
        _('GLCM Entropy'), _('GLCM Variance'), _('GLCM Sum Average'), _('GLCM Sum Variance'),
        _('GLCM Sum Entropy'), _('GLCM Diff Average'), _('GLCM Diff Variance'), _('GLCM Diff Entropy')
    ])
    
    # Add first-order statistical features to header
    header.extend([
        _('Intensity Mean'), _('Intensity Std'), _('Intensity Variance'), _('Intensity Skewness'),
        _('Intensity Kurtosis'), _('Intensity Min'), _('Intensity Max'), _('Intensity Range'),
        _('Intensity P10'), _('Intensity P25'), _('Intensity P75'), _('Intensity P90'),
        _('Intensity IQR'), _('Intensity Entropy'), _('Intensity Energy'), _('Intensity Median'),
        _('Intensity MAD'), _('Intensity CV')
    ])
    
    if analysis.cell.scale_set:
        header.extend([
            _('Area (μm²)'), _('Perimeter (μm)'), _('Major Axis (μm)'), _('Minor Axis (μm)'),
            _('Scale (px/μm)')
        ])
    
    writer.writerow(header)
    
    # Write data
    for cell in analysis.detected_cells.all():
        row = [
            cell.cell_id, cell.area, cell.perimeter, cell.circularity, cell.eccentricity,
            cell.solidity, cell.extent, cell.major_axis_length, cell.minor_axis_length,
            cell.aspect_ratio, cell.centroid_x, cell.centroid_y, cell.bounding_box_x,
            cell.bounding_box_y, cell.bounding_box_width, cell.bounding_box_height
        ]
        
        # Add GLCM texture features to row
        row.extend([
            cell.glcm_contrast or '', cell.glcm_correlation or '', cell.glcm_energy or '',
            cell.glcm_homogeneity or '', cell.glcm_entropy or '', cell.glcm_variance or '',
            cell.glcm_sum_average or '', cell.glcm_sum_variance or '', cell.glcm_sum_entropy or '',
            cell.glcm_difference_average or '', cell.glcm_difference_variance or '', 
            cell.glcm_difference_entropy or ''
        ])
        
        # Add first-order statistical features to row
        row.extend([
            cell.intensity_mean or '', cell.intensity_std or '', cell.intensity_variance or '',
            cell.intensity_skewness or '', cell.intensity_kurtosis or '', cell.intensity_min or '',
            cell.intensity_max or '', cell.intensity_range or '', cell.intensity_p10 or '',
            cell.intensity_p25 or '', cell.intensity_p75 or '', cell.intensity_p90 or '',
            cell.intensity_iqr or '', cell.intensity_entropy or '', cell.intensity_energy or '',
            cell.intensity_median or '', cell.intensity_mad or '', cell.intensity_cv or ''
        ])
        
        if analysis.cell.scale_set:
            row.extend([
                cell.area_microns_sq or '', cell.perimeter_microns or '', 
                cell.major_axis_length_microns or '', cell.minor_axis_length_microns or '',
                analysis.cell.pixels_per_micron
            ])
        
        writer.writerow(row)
    
    return response


@login_required
def delete_analysis(request, analysis_id):
    """Delete an analysis"""
    analysis = get_object_or_404(CellAnalysis, id=analysis_id, cell__user=request.user)
    
    if request.method == 'POST':
        cell_id = analysis.cell.id
        analysis.delete()
        messages.success(request, _('Analysis deleted successfully.'))
        return redirect('cells:cell_detail', cell_id=cell_id)
    
    return redirect('cells:analysis_detail', analysis_id=analysis_id)


@login_required
def set_scale_calibration(request, cell_id):
    """Set scale calibration for a cell image"""
    cell = get_object_or_404(Cell, id=cell_id, user=request.user)
    
    if request.method == 'POST':
        form = ScaleCalibrationForm(request.POST)
        if form.is_valid():
            pixels = form.cleaned_data['reference_length_pixels']
            microns = form.cleaned_data['reference_length_microns']
            
            cell.set_scale_calibration(pixels, microns)
            
            messages.success(
                request, 
                _('Scale calibration set successfully! Scale: {:.2f} pixels/μm').format(cell.pixels_per_micron)
            )
            return redirect('cells:cell_detail', cell_id=cell.id)
    else:
        # Pre-populate with existing values if available
        initial_data = {}
        if cell.scale_set:
            initial_data = {
                'reference_length_pixels': cell.scale_reference_length_pixels,
                'reference_length_microns': cell.scale_reference_length_microns
            }
        form = ScaleCalibrationForm(initial=initial_data)
    
    context = {
        'cell': cell,
        'form': form,
    }
    return render(request, 'cells/set_scale.html', context)
