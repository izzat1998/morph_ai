from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.utils.translation import gettext as _
import csv
import json
from .forms import CellUploadForm, CellAnalysisForm, ScaleCalibrationForm
from .models import Cell, CellAnalysis, DetectedCell
from .analysis import run_cell_analysis, get_analysis_summary, get_image_quality_summary


@login_required
def upload_cell(request):
    if request.method == 'POST':
        form = CellUploadForm(request.POST, request.FILES)
        if form.is_valid():
            cell = form.save(commit=False)
            cell.user = request.user
            cell.save()
            messages.success(request, _('Cell image "%(name)s" uploaded successfully!') % {'name': cell.name})
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
            analysis = form.save(commit=False)
            analysis.cell = cell
            
            # Handle ROI data if provided
            print(f"DEBUG VIEW: use_roi = {analysis.use_roi}")
            print(f"DEBUG VIEW: 'roi_data' in POST: {'roi_data' in request.POST}")
            
            if analysis.use_roi and 'roi_data' in request.POST:
                try:
                    roi_data_str = request.POST.get('roi_data', '[]')
                    print(f"DEBUG VIEW: roi_data_str = {roi_data_str}")
                    
                    roi_data = json.loads(roi_data_str)
                    roi_count = int(request.POST.get('roi_count', '0'))
                    
                    print(f"DEBUG VIEW: parsed roi_data = {roi_data}")
                    print(f"DEBUG VIEW: roi_count = {roi_count}")
                    
                    analysis.roi_regions = roi_data
                    analysis.roi_count = roi_count
                    
                    messages.success(request, _('ROI selection enabled with {} region(s)').format(roi_count))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"DEBUG VIEW: JSON decode error: {e}")
                    messages.warning(request, _('ROI data was invalid, proceeding without ROI selection'))
                    analysis.use_roi = False
                    analysis.roi_regions = []
                    analysis.roi_count = 0
            elif analysis.use_roi:
                print("DEBUG VIEW: ROI enabled but no roi_data in POST")
                messages.warning(request, _('ROI selection was enabled but no regions were drawn. Running standard analysis.'))
                analysis.use_roi = False
            
            analysis.save()
            
            # Run analysis in the background (for now, synchronously)
            # In production, this should be done with Celery or similar
            success = run_cell_analysis(analysis.id)
            
            if success:
                messages.success(request, _('Analysis completed successfully!'))
                return redirect('cells:analysis_detail', analysis_id=analysis.id)
            else:
                messages.error(request, _('Analysis failed. Please check the error details.'))
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
    
    context = {
        'analysis': analysis,
        'cell': analysis.cell,
        'summary': get_analysis_summary(analysis) if analysis.status == 'completed' else None,
    }
    
    if analysis.status == 'completed':
        # Paginate detected cells
        detected_cells = analysis.detected_cells.all()
        paginator = Paginator(detected_cells, 20)  # Show 20 cells per page
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context['page_obj'] = page_obj
        
        # Add cell filtering information for transparency
        validated_cell_count = detected_cells.count()
        context['validated_cell_count'] = validated_cell_count
        context['original_cell_count'] = analysis.num_cells_detected
        context['cells_filtered'] = analysis.num_cells_detected - validated_cell_count
        
        # Extract filtering details from quality metrics
        filtering_info = {}
        if analysis.quality_metrics:
            # Segmentation refinement info
            if 'segmentation_refinement' in analysis.quality_metrics:
                refinement = analysis.quality_metrics['segmentation_refinement']
                filtering_info['refinement'] = {
                    'original_count': refinement.get('original_cell_count', analysis.num_cells_detected),
                    'refined_count': refinement.get('refined_cell_count', analysis.num_cells_detected),
                    'steps': refinement.get('refinement_steps', [])
                }
            
            # Morphometric validation info  
            if 'morphometric_validation' in analysis.quality_metrics:
                validation = analysis.quality_metrics['morphometric_validation']
                filtering_info['validation'] = {
                    'total_cells': validation.get('total_cells', 0),
                    'valid_cells': validation.get('valid_cells', 0),
                    'outliers_detected': validation.get('outliers_detected', 0),
                    'outlier_percentage': validation.get('outlier_percentage', 0),
                    'outlier_reasons': validation.get('outlier_reasons', {})
                }
        
        context['filtering_info'] = filtering_info
    
    return render(request, 'cells/analysis_detail.html', context)


@login_required
def analysis_list(request):
    analyses = CellAnalysis.objects.filter(cell__user=request.user).order_by('-analysis_date')
    
    # Filter by status if provided
    status_filter = request.GET.get('status')
    if status_filter:
        analyses = analyses.filter(status=status_filter)
    
    # Paginate results
    paginator = Paginator(analyses, 10)  # Show 10 analyses per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'status_choices': CellAnalysis.STATUS_CHOICES,
    }
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
