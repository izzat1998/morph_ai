from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
import csv
import json
import logging
from .forms import CellUploadForm, CellAnalysisForm, ScaleCalibrationForm, BatchCreateForm, BatchAnalysisForm
from .models import Cell, CellAnalysis, DetectedCell, AnalysisBatch
from .analysis import run_cell_analysis, get_analysis_summary, get_image_quality_summary
from .services import CellUploadService, CellAnalysisService, CellExportService, CellScaleService
# Import enhanced analysis for statistical features
from .enhanced_analysis import EnhancedMorphometricAnalysis
# Import visualization helpers
from .visualization_helpers import prepare_analysis_context_enhanced
from django.utils.translation import gettext as _

logger = logging.getLogger(__name__)

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

            # Check if statistical analysis is enabled
            statistical_config = analysis.preprocessing_options.get('statistical_config', {}) if analysis.preprocessing_options else {}
            enable_statistical = statistical_config.get('enable_statistical_analysis', False)

            # Run analysis in the background (for now, synchronously)
            # In production, this should be done with Celery or similar
            if enable_statistical:
                # Use enhanced analysis with statistical features
                try:
                    enhanced_analyzer = EnhancedMorphometricAnalysis(
                        analysis_id=analysis.id,
                        enable_statistics=True
                    )
                    success = enhanced_analyzer.run_enhanced_analysis()
                    if success:
                        messages.success(request, 'Расширенный статистический анализ завершен успешно!')
                    else:
                        messages.error(request, 'Ошибка статистического анализа. Переключение на стандартный анализ...')
                        # Fallback to standard analysis
                        success = CellAnalysisService.run_analysis(analysis.id)
                except Exception as e:
                    messages.error(request, f'Ошибка статистического анализа: {str(e)}. Переключение на стандартный анализ...')
                    # Fallback to standard analysis
                    success = CellAnalysisService.run_analysis(analysis.id)
            else:
                # Use standard analysis (backward compatible)
                success = CellAnalysisService.run_analysis(analysis.id)

            if success:
                if not enable_statistical:
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
    
    # Add statistical analysis data if available
    try:
        from morphometric_stats.models import StatisticalAnalysis, FeatureStatistics
        
        statistical_analysis = getattr(analysis, 'statistical_analysis', None)
        if statistical_analysis:
            # Get statistical summary data
            feature_stats = FeatureStatistics.objects.filter(
                statistical_analysis=statistical_analysis
            ).select_related('detected_cell')
            
            # Organize statistical data by feature type
            statistical_summary = {}
            for stat in feature_stats:
                if stat.detected_cell is None:  # Population-level statistics
                    statistical_summary[stat.feature_name] = {
                        'mean': stat.mean_value,
                        'std_error': stat.std_error,
                        'confidence_interval_lower': stat.confidence_interval_lower,
                        'confidence_interval_upper': stat.confidence_interval_upper,
                        'confidence_interval_width': stat.confidence_interval_width,
                        'uncertainty_percent': stat.uncertainty_percent,
                        'reliability_score': stat.measurement_reliability_score,
                    }
            
            # Add formatted confidence level for display
            statistical_analysis_display = {
                'confidence_level': statistical_analysis.confidence_level,
                'confidence_level_percent': int(statistical_analysis.confidence_level * 100),
                'bootstrap_iterations': statistical_analysis.bootstrap_iterations,
                'pixel_uncertainty': statistical_analysis.pixel_uncertainty,
                'computation_time_seconds': statistical_analysis.computation_time_seconds,
            }
            
            # Get and format statistical config for display
            statistical_config = analysis.preprocessing_options.get('statistical_config', {}) if analysis.preprocessing_options else {}
            statistical_config_display = statistical_config.copy()
            if 'confidence_level' in statistical_config_display:
                statistical_config_display['confidence_level_percent'] = int(statistical_config_display['confidence_level'] * 100)
            
            context.update({
                'statistical_analysis': statistical_analysis,
                'statistical_analysis_display': statistical_analysis_display,
                'statistical_summary': statistical_summary,
                'has_statistical_data': True,
                'statistical_config': statistical_config,
                'statistical_config_display': statistical_config_display,
            })
        else:
            # Get and format statistical config for display even if no analysis data
            statistical_config = analysis.preprocessing_options.get('statistical_config', {}) if analysis.preprocessing_options else {}
            statistical_config_display = statistical_config.copy()
            if 'confidence_level' in statistical_config_display:
                statistical_config_display['confidence_level_percent'] = int(statistical_config_display['confidence_level'] * 100)
            
            context.update({
                'has_statistical_data': False,
                'statistical_config': statistical_config,
                'statistical_config_display': statistical_config_display,
            })
    except ImportError:
        # Statistical models not available
        context.update({
            'has_statistical_data': False,
            'statistical_config': {},
        })
    
    # Add enhanced visualization data
    enhanced_context = prepare_analysis_context_enhanced(analysis, request)
    context.update(enhanced_context)
    
    return render(request, 'cells/analysis_detail.html', context)


@login_required
def enhanced_analysis_detail(request, analysis_id):
    """
    Enhanced analysis detail view with professional visualization
    """
    analysis = get_object_or_404(CellAnalysis, id=analysis_id, cell__user=request.user)
    context = CellAnalysisService.get_analysis_context(analysis, request)
    
    # Add statistical analysis data if available
    try:
        from morphometric_stats.models import StatisticalAnalysis, FeatureStatistics
        
        statistical_analysis = getattr(analysis, 'statistical_analysis', None)
        if statistical_analysis:
            # Get statistical summary data
            feature_stats = FeatureStatistics.objects.filter(
                statistical_analysis=statistical_analysis
            ).select_related('detected_cell')
            
            # Organize statistical data by feature type
            statistical_summary = {}
            for stat in feature_stats:
                if stat.detected_cell is None:  # Population-level statistics
                    statistical_summary[stat.feature_name] = {
                        'mean': stat.mean_value,
                        'std_error': stat.std_error,
                        'confidence_interval_lower': stat.confidence_interval_lower,
                        'confidence_interval_upper': stat.confidence_interval_upper,
                        'confidence_interval_width': stat.confidence_interval_width,
                        'uncertainty_percent': stat.uncertainty_percent,
                        'reliability_score': stat.measurement_reliability_score,
                    }
            
            # Add formatted confidence level for display
            statistical_analysis_display = {
                'confidence_level': statistical_analysis.confidence_level,
                'confidence_level_percent': int(statistical_analysis.confidence_level * 100),
                'bootstrap_iterations': statistical_analysis.bootstrap_iterations,
                'pixel_uncertainty': statistical_analysis.pixel_uncertainty,
                'computation_time_seconds': statistical_analysis.computation_time_seconds,
            }
            
            # Get and format statistical config for display
            statistical_config = analysis.preprocessing_options.get('statistical_config', {}) if analysis.preprocessing_options else {}
            statistical_config_display = statistical_config.copy()
            if 'confidence_level' in statistical_config_display:
                statistical_config_display['confidence_level_percent'] = int(statistical_config_display['confidence_level'] * 100)
            
            context.update({
                'statistical_analysis': statistical_analysis,
                'statistical_analysis_display': statistical_analysis_display,
                'statistical_summary': statistical_summary,
                'has_statistical_data': True,
                'statistical_config': statistical_config,
                'statistical_config_display': statistical_config_display,
            })
        else:
            # Get and format statistical config for display even if no analysis data
            statistical_config = analysis.preprocessing_options.get('statistical_config', {}) if analysis.preprocessing_options else {}
            statistical_config_display = statistical_config.copy()
            if 'confidence_level' in statistical_config_display:
                statistical_config_display['confidence_level_percent'] = int(statistical_config_display['confidence_level'] * 100)
            
            context.update({
                'has_statistical_data': False,
                'statistical_config': statistical_config,
                'statistical_config_display': statistical_config_display,
            })
    except ImportError:
        # Statistical models not available
        context.update({
            'has_statistical_data': False,
            'statistical_config': {},
        })
    
    # Add enhanced visualization data
    enhanced_context = prepare_analysis_context_enhanced(analysis, request)
    context.update(enhanced_context)
    
    return render(request, 'cells/enhanced_analysis_detail.html', context)


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


@login_required
def optimize_analysis_parameters(request, cell_id):
    """
    AJAX endpoint for automatic parameter optimization based on image analysis.
    Returns optimized parameters as JSON for real-time UI updates.
    """
    cell = get_object_or_404(Cell, id=cell_id, user=request.user)
    
    try:
        from PIL import Image as PILImage
        import numpy as np
        from .quality_assessment import ImageQualityAssessment
        from .parameter_optimization import ParameterOptimizer
        import io
        
        # Load and convert image for analysis
        with cell.image.open('rb') as img_file:
            pil_image = PILImage.open(img_file)
            
            # Convert to numpy array
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
        
        # Perform quality assessment
        quality_metrics = ImageQualityAssessment.assess_overall_quality(image_array)
        
        # Optimize parameters
        optimized_params = ParameterOptimizer.optimize_all_parameters(
            image_array, quality_metrics
        )
        
        # Create response with optimization results
        response_data = {
            'success': True,
            'optimized_parameters': {
                'cellpose_model': optimized_params['cellpose_model'],
                'cellpose_diameter': round(optimized_params['cellpose_diameter'], 1),
                'flow_threshold': round(optimized_params['flow_threshold'], 2),
                'cellprob_threshold': round(optimized_params['cellprob_threshold'], 2),
            },
            'quality_assessment': {
                'overall_score': round(quality_metrics['overall_score'], 1),
                'blur_score': round(quality_metrics['blur_score'], 1),
                'contrast_score': round(quality_metrics['contrast_score'], 1),
                'noise_score': round(quality_metrics['noise_score'], 1),
                'quality_category': quality_metrics['quality_category'],
                'recommendations': quality_metrics['recommendations']
            },
            'confidence_scores': optimized_params['confidence_scores'],
            'optimization_notes': optimized_params['optimization_notes'],
            'overall_confidence': round(optimized_params['overall_confidence'], 2)
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Parameter optimization failed for cell {cell_id}: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'fallback_parameters': {
                'cellpose_model': 'cyto2',
                'cellpose_diameter': 30.0,
                'flow_threshold': 0.4,
                'cellprob_threshold': 0.0,
            }
        })


# ===============================================
# BATCH PROCESSING VIEWS (RTX 2070 Max-Q Optimized)
# ===============================================

@login_required
def batch_list(request):
    """List all analysis batches for the current user"""
    batches = AnalysisBatch.objects.filter(user=request.user).order_by('-created_at')
    
    # Add pagination for large numbers of batches
    paginator = Paginator(batches, 10)  # Show 10 batches per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'batches': page_obj,
        'page_obj': page_obj,
    }
    return render(request, 'cells/batch_list.html', context)


@login_required
def batch_create(request):
    """Create a new analysis batch with multiple images"""
    if request.method == 'POST':
        form = BatchCreateForm(request.POST, request.FILES)
        if form.is_valid():
            # Create the batch
            batch = form.save(commit=False)
            batch.user = request.user
            batch.save()
            
            # Process uploaded images
            images = form.cleaned_data['images']
            batch.total_images = len(images)
            batch.save()
            
            # Create Cell objects for each uploaded image
            created_cells = []
            for i, image_file in enumerate(images):
                # Generate a name for the cell image
                base_name = image_file.name.rsplit('.', 1)[0]  # Remove extension
                cell_name = f"{batch.name} - Image {i+1} ({base_name})"
                
                # Create Cell object
                cell = Cell.objects.create(
                    user=request.user,
                    name=cell_name,
                    image=image_file,
                    batch=batch
                )
                created_cells.append(cell)
            
            messages.success(
                request, 
                f'Пакет "{batch.name}" создан с {len(created_cells)} изображениями. '
                f'Теперь настройте параметры анализа.'
            )
            return redirect('cells:batch_configure', batch_id=batch.id)
    else:
        form = BatchCreateForm()
    
    context = {
        'form': form,
    }
    return render(request, 'cells/batch_create.html', context)


@login_required
def batch_configure(request, batch_id):
    """Configure analysis parameters for a batch"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    # Only allow configuration if batch is pending
    if batch.status != 'pending':
        messages.error(request, 'Этот пакет уже обрабатывается или завершен.')
        return redirect('cells:batch_detail', batch_id=batch.id)
    
    if request.method == 'POST':
        form = BatchAnalysisForm(request.POST)
        if form.is_valid():
            # Store analysis parameters in the batch
            batch.analysis_parameters = form.get_analysis_parameters()
            batch.save()
            
            # Start batch processing
            try:
                from .services import BatchProcessingService
                
                # Start processing in the background (simplified for now)
                success = BatchProcessingService.start_batch_processing(batch.id)
                
                if success:
                    messages.success(
                        request, 
                        f'Пакетный анализ запущен! Обработка {batch.total_images} изображений началась.'
                    )
                else:
                    messages.error(request, 'Ошибка запуска пакетного анализа.')
                
            except ImportError:
                messages.error(request, 'Сервис пакетной обработки недоступен.')
            
            return redirect('cells:batch_detail', batch_id=batch.id)
    else:
        form = BatchAnalysisForm()
    
    context = {
        'batch': batch,
        'form': form,
        'batch_images': batch.cells.all()[:5],  # Show first 5 images as preview
    }
    return render(request, 'cells/batch_configure.html', context)


@login_required
def batch_detail(request, batch_id):
    """Display batch analysis details and results"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    # Get all cells in the batch with their analyses
    batch_cells = batch.cells.prefetch_related('analyses__detected_cells').all()
    
    # Calculate batch statistics if completed
    batch_stats = None
    if batch.status == 'completed':
        batch_stats = calculate_batch_statistics(batch)
    
    context = {
        'batch': batch,
        'batch_cells': batch_cells,
        'batch_stats': batch_stats,
        'progress_percentage': batch.progress_percentage,
    }
    return render(request, 'cells/batch_detail.html', context)


@login_required
def batch_progress(request, batch_id):
    """AJAX endpoint for real-time batch progress updates"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    # Get processing status for each image
    image_status = []
    for cell in batch.cells.all():
        latest_analysis = cell.analyses.filter(status__in=['completed', 'failed']).first()
        status = 'pending'
        if latest_analysis:
            status = latest_analysis.status
        
        image_status.append({
            'cell_id': cell.id,
            'cell_name': cell.name,
            'status': status,
            'num_cells_detected': latest_analysis.num_cells_detected if latest_analysis else 0
        })
    
    return JsonResponse({
        'batch_status': batch.status,
        'progress_percentage': batch.progress_percentage,
        'processed_images': batch.processed_images,
        'total_images': batch.total_images,
        'failed_images': batch.failed_images,
        'image_status': image_status,
        'error_message': batch.error_message,
    })


@login_required
def batch_cancel(request, batch_id):
    """Cancel batch processing"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    if request.method == 'POST' and batch.status == 'processing':
        batch.status = 'cancelled'
        batch.save()
        messages.success(request, 'Пакетный анализ отменен.')
    
    return redirect('cells:batch_detail', batch_id=batch.id)


@login_required
def batch_delete(request, batch_id):
    """Delete a batch and all associated data"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    if request.method == 'POST':
        batch_name = batch.name
        
        # Delete all associated cells and their analyses
        batch.cells.all().delete()
        
        # Delete the batch
        batch.delete()
        
        messages.success(request, f'Пакет "{batch_name}" и все связанные данные удалены.')
        return redirect('cells:batch_list')
    
    context = {
        'batch': batch,
    }
    return render(request, 'cells/batch_delete_confirm.html', context)


def calculate_batch_statistics(batch):
    """Calculate comprehensive statistics for a completed batch including all morphometric features"""
    if batch.status != 'completed':
        return None
    
    # Get all completed analyses in the batch
    completed_analyses = CellAnalysis.objects.filter(
        cell__batch=batch,
        status='completed'
    ).prefetch_related('detected_cells')
    
    if not completed_analyses.exists():
        return None
    
    # Initialize data collectors for all morphometric features
    morphometric_data = {
        # Basic morphometric features
        'areas': [],
        'perimeters': [],
        'circularities': [],
        'eccentricities': [],
        'solidities': [],
        'extents': [],
        'major_axis_lengths': [],
        'minor_axis_lengths': [],
        'aspect_ratios': [],
        
        # GLCM texture features
        'glcm_contrasts': [],
        'glcm_correlations': [],
        'glcm_energies': [],
        'glcm_homogeneities': [],
        'glcm_entropies': [],
        'glcm_variances': [],
        'glcm_sum_averages': [],
        'glcm_sum_variances': [],
        'glcm_sum_entropies': [],
        'glcm_diff_averages': [],
        'glcm_diff_variances': [],
        'glcm_diff_entropies': [],
        
        # Intensity features
        'intensity_means': [],
        'intensity_stds': [],
        'intensity_variances': [],
        'intensity_skewnesses': [],
        'intensity_kurtoses': [],
        'intensity_mins': [],
        'intensity_maxs': [],
        'intensity_ranges': [],
        'intensity_p10s': [],
        'intensity_p25s': [],
        'intensity_p75s': [],
        'intensity_p90s': [],
        'intensity_iqrs': [],
        'intensity_entropies': [],
        'intensity_energies': [],
        'intensity_medians': [],
        'intensity_mads': [],
        'intensity_cvs': [],
        
        # Physical measurements (if scale available)
        'areas_microns': [],
        'perimeters_microns': [],
        'major_axis_lengths_microns': [],
        'minor_axis_lengths_microns': [],
    }
    
    total_cells = 0
    scale_available = False
    
    # Collect data from all detected cells across all analyses
    for analysis in completed_analyses:
        detected_cells = analysis.detected_cells.all()
        total_cells += len(detected_cells)
        
        # Check if scale is available for any analysis
        if analysis.cell.scale_set:
            scale_available = True
        
        for cell in detected_cells:
            # Basic morphometric features
            if cell.area is not None:
                morphometric_data['areas'].append(cell.area)
            if cell.perimeter is not None:
                morphometric_data['perimeters'].append(cell.perimeter)
            if cell.circularity is not None:
                morphometric_data['circularities'].append(cell.circularity)
            if cell.eccentricity is not None:
                morphometric_data['eccentricities'].append(cell.eccentricity)
            if cell.solidity is not None:
                morphometric_data['solidities'].append(cell.solidity)
            if cell.extent is not None:
                morphometric_data['extents'].append(cell.extent)
            if cell.major_axis_length is not None:
                morphometric_data['major_axis_lengths'].append(cell.major_axis_length)
            if cell.minor_axis_length is not None:
                morphometric_data['minor_axis_lengths'].append(cell.minor_axis_length)
            if cell.aspect_ratio is not None:
                morphometric_data['aspect_ratios'].append(cell.aspect_ratio)
            
            # GLCM texture features
            if cell.glcm_contrast is not None:
                morphometric_data['glcm_contrasts'].append(cell.glcm_contrast)
            if cell.glcm_correlation is not None:
                morphometric_data['glcm_correlations'].append(cell.glcm_correlation)
            if cell.glcm_energy is not None:
                morphometric_data['glcm_energies'].append(cell.glcm_energy)
            if cell.glcm_homogeneity is not None:
                morphometric_data['glcm_homogeneities'].append(cell.glcm_homogeneity)
            if cell.glcm_entropy is not None:
                morphometric_data['glcm_entropies'].append(cell.glcm_entropy)
            if cell.glcm_variance is not None:
                morphometric_data['glcm_variances'].append(cell.glcm_variance)
            if cell.glcm_sum_average is not None:
                morphometric_data['glcm_sum_averages'].append(cell.glcm_sum_average)
            if cell.glcm_sum_variance is not None:
                morphometric_data['glcm_sum_variances'].append(cell.glcm_sum_variance)
            if cell.glcm_sum_entropy is not None:
                morphometric_data['glcm_sum_entropies'].append(cell.glcm_sum_entropy)
            if cell.glcm_difference_average is not None:
                morphometric_data['glcm_diff_averages'].append(cell.glcm_difference_average)
            if cell.glcm_difference_variance is not None:
                morphometric_data['glcm_diff_variances'].append(cell.glcm_difference_variance)
            if cell.glcm_difference_entropy is not None:
                morphometric_data['glcm_diff_entropies'].append(cell.glcm_difference_entropy)
            
            # Intensity features
            if cell.intensity_mean is not None:
                morphometric_data['intensity_means'].append(cell.intensity_mean)
            if cell.intensity_std is not None:
                morphometric_data['intensity_stds'].append(cell.intensity_std)
            if cell.intensity_variance is not None:
                morphometric_data['intensity_variances'].append(cell.intensity_variance)
            if cell.intensity_skewness is not None:
                morphometric_data['intensity_skewnesses'].append(cell.intensity_skewness)
            if cell.intensity_kurtosis is not None:
                morphometric_data['intensity_kurtoses'].append(cell.intensity_kurtosis)
            if cell.intensity_min is not None:
                morphometric_data['intensity_mins'].append(cell.intensity_min)
            if cell.intensity_max is not None:
                morphometric_data['intensity_maxs'].append(cell.intensity_max)
            if cell.intensity_range is not None:
                morphometric_data['intensity_ranges'].append(cell.intensity_range)
            if cell.intensity_p10 is not None:
                morphometric_data['intensity_p10s'].append(cell.intensity_p10)
            if cell.intensity_p25 is not None:
                morphometric_data['intensity_p25s'].append(cell.intensity_p25)
            if cell.intensity_p75 is not None:
                morphometric_data['intensity_p75s'].append(cell.intensity_p75)
            if cell.intensity_p90 is not None:
                morphometric_data['intensity_p90s'].append(cell.intensity_p90)
            if cell.intensity_iqr is not None:
                morphometric_data['intensity_iqrs'].append(cell.intensity_iqr)
            if cell.intensity_entropy is not None:
                morphometric_data['intensity_entropies'].append(cell.intensity_entropy)
            if cell.intensity_energy is not None:
                morphometric_data['intensity_energies'].append(cell.intensity_energy)
            if cell.intensity_median is not None:
                morphometric_data['intensity_medians'].append(cell.intensity_median)
            if cell.intensity_mad is not None:
                morphometric_data['intensity_mads'].append(cell.intensity_mad)
            if cell.intensity_cv is not None:
                morphometric_data['intensity_cvs'].append(cell.intensity_cv)
            
            # Physical measurements (if scale is available)
            if scale_available and analysis.cell.scale_set:
                if cell.area_microns_sq is not None:
                    morphometric_data['areas_microns'].append(cell.area_microns_sq)
                if cell.perimeter_microns is not None:
                    morphometric_data['perimeters_microns'].append(cell.perimeter_microns)
                if cell.major_axis_length_microns is not None:
                    morphometric_data['major_axis_lengths_microns'].append(cell.major_axis_length_microns)
                if cell.minor_axis_length_microns is not None:
                    morphometric_data['minor_axis_lengths_microns'].append(cell.minor_axis_length_microns)
    
    # If no valid data found, return None
    if not morphometric_data['areas']:
        return None
    
    import statistics
    import numpy as np
    
    def calculate_feature_stats(values, feature_name):
        """Calculate comprehensive statistics for a feature"""
        if not values:
            return None
        
        try:
            return {
                'count': len(values),
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values),
                'q25': np.percentile(values, 25) if len(values) > 1 else values[0],
                'q75': np.percentile(values, 75) if len(values) > 1 else values[0],
                'iqr': np.percentile(values, 75) - np.percentile(values, 25) if len(values) > 1 else 0,
                'range': max(values) - min(values),
                'cv': (statistics.stdev(values) / statistics.mean(values)) * 100 if len(values) > 1 and statistics.mean(values) != 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error calculating stats for {feature_name}: {str(e)}")
            return None
    
    # Build comprehensive statistics
    stats = {
        'total_cells_across_all_images': total_cells,
        'total_images_analyzed': completed_analyses.count(),
        'average_cells_per_image': total_cells / completed_analyses.count() if completed_analyses.count() > 0 else 0,
        'scale_available': scale_available,
        
        # Basic morphometric features
        'area_stats': calculate_feature_stats(morphometric_data['areas'], 'area'),
        'perimeter_stats': calculate_feature_stats(morphometric_data['perimeters'], 'perimeter'),
        'circularity_stats': calculate_feature_stats(morphometric_data['circularities'], 'circularity'),
        'eccentricity_stats': calculate_feature_stats(morphometric_data['eccentricities'], 'eccentricity'),
        'solidity_stats': calculate_feature_stats(morphometric_data['solidities'], 'solidity'),
        'extent_stats': calculate_feature_stats(morphometric_data['extents'], 'extent'),
        'major_axis_stats': calculate_feature_stats(morphometric_data['major_axis_lengths'], 'major_axis'),
        'minor_axis_stats': calculate_feature_stats(morphometric_data['minor_axis_lengths'], 'minor_axis'),
        'aspect_ratio_stats': calculate_feature_stats(morphometric_data['aspect_ratios'], 'aspect_ratio'),
        
        # GLCM texture features
        'glcm_contrast_stats': calculate_feature_stats(morphometric_data['glcm_contrasts'], 'glcm_contrast'),
        'glcm_correlation_stats': calculate_feature_stats(morphometric_data['glcm_correlations'], 'glcm_correlation'),
        'glcm_energy_stats': calculate_feature_stats(morphometric_data['glcm_energies'], 'glcm_energy'),
        'glcm_homogeneity_stats': calculate_feature_stats(morphometric_data['glcm_homogeneities'], 'glcm_homogeneity'),
        'glcm_entropy_stats': calculate_feature_stats(morphometric_data['glcm_entropies'], 'glcm_entropy'),
        'glcm_variance_stats': calculate_feature_stats(morphometric_data['glcm_variances'], 'glcm_variance'),
        
        # Intensity features (key ones)
        'intensity_mean_stats': calculate_feature_stats(morphometric_data['intensity_means'], 'intensity_mean'),
        'intensity_std_stats': calculate_feature_stats(morphometric_data['intensity_stds'], 'intensity_std'),
        'intensity_range_stats': calculate_feature_stats(morphometric_data['intensity_ranges'], 'intensity_range'),
        'intensity_entropy_stats': calculate_feature_stats(morphometric_data['intensity_entropies'], 'intensity_entropy'),
    }
    
    # Add physical measurements if scale is available
    if scale_available:
        stats.update({
            'area_stats_microns': calculate_feature_stats(morphometric_data['areas_microns'], 'area_microns'),
            'perimeter_stats_microns': calculate_feature_stats(morphometric_data['perimeters_microns'], 'perimeter_microns'),
            'major_axis_stats_microns': calculate_feature_stats(morphometric_data['major_axis_lengths_microns'], 'major_axis_microns'),
            'minor_axis_stats_microns': calculate_feature_stats(morphometric_data['minor_axis_lengths_microns'], 'minor_axis_microns'),
        })
    
    # Add cross-analysis comparison metrics
    stats['analysis_comparison'] = _calculate_cross_analysis_metrics(completed_analyses)
    
    return stats


def _calculate_cross_analysis_metrics(completed_analyses):
    """Calculate comparison metrics between different analyses in a batch"""
    try:
        analysis_metrics = []
        
        for analysis in completed_analyses:
            detected_cells = analysis.detected_cells.all()
            cell_count = detected_cells.count()
            
            if cell_count > 0:
                # Calculate metrics for this analysis
                areas = [cell.area for cell in detected_cells if cell.area is not None]
                circularities = [cell.circularity for cell in detected_cells if cell.circularity is not None]
                
                if areas and circularities:
                    import statistics
                    analysis_metric = {
                        'analysis_id': analysis.id,
                        'cell_name': analysis.cell.name,
                        'cell_count': cell_count,
                        'mean_area': statistics.mean(areas),
                        'mean_circularity': statistics.mean(circularities),
                        'processing_time': analysis.processing_time or 0,
                    }
                    analysis_metrics.append(analysis_metric)
        
        if not analysis_metrics:
            return None
        
        # Calculate comparison statistics
        cell_counts = [m['cell_count'] for m in analysis_metrics]
        mean_areas = [m['mean_area'] for m in analysis_metrics]
        mean_circularities = [m['mean_circularity'] for m in analysis_metrics]
        processing_times = [m['processing_time'] for m in analysis_metrics]
        
        import statistics
        import numpy as np
        
        comparison_stats = {
            'individual_analyses': analysis_metrics,
            'cross_analysis_metrics': {
                'cell_count_variation': {
                    'mean': statistics.mean(cell_counts),
                    'std': statistics.stdev(cell_counts) if len(cell_counts) > 1 else 0,
                    'cv': (statistics.stdev(cell_counts) / statistics.mean(cell_counts)) * 100 if len(cell_counts) > 1 and statistics.mean(cell_counts) != 0 else 0,
                    'range': max(cell_counts) - min(cell_counts)
                },
                'area_consistency': {
                    'mean_of_means': statistics.mean(mean_areas),
                    'std_of_means': statistics.stdev(mean_areas) if len(mean_areas) > 1 else 0,
                    'cv_between_images': (statistics.stdev(mean_areas) / statistics.mean(mean_areas)) * 100 if len(mean_areas) > 1 and statistics.mean(mean_areas) != 0 else 0
                },
                'circularity_consistency': {
                    'mean_of_means': statistics.mean(mean_circularities),
                    'std_of_means': statistics.stdev(mean_circularities) if len(mean_circularities) > 1 else 0,
                    'cv_between_images': (statistics.stdev(mean_circularities) / statistics.mean(mean_circularities)) * 100 if len(mean_circularities) > 1 and statistics.mean(mean_circularities) != 0 else 0
                },
                'processing_efficiency': {
                    'mean_processing_time': statistics.mean(processing_times),
                    'total_processing_time': sum(processing_times),
                    'fastest_analysis': min(processing_times),
                    'slowest_analysis': max(processing_times)
                }
            }
        }
        
        return comparison_stats
        
    except Exception as e:
        logger.error(f"Error calculating cross-analysis metrics: {str(e)}")
        return None


@login_required
def batch_statistics_json(request, batch_id):
    """JSON API endpoint for comprehensive batch statistics data"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    if batch.status != 'completed':
        return JsonResponse({
            'error': 'Batch analysis not completed',
            'status': batch.status
        }, status=400)
    
    try:
        batch_stats = calculate_batch_statistics(batch)
        if not batch_stats:
            return JsonResponse({
                'error': 'No statistics available for this batch',
                'status': batch.status
            }, status=404)
        
        return JsonResponse({
            'success': True,
            'batch_id': batch.id,
            'batch_name': batch.name,
            'batch_status': batch.status,
            'statistics': batch_stats
        })
        
    except Exception as e:
        logger.error(f"Error generating batch statistics JSON for batch {batch_id}: {str(e)}")
        return JsonResponse({
            'error': f'Failed to generate statistics: {str(e)}'
        }, status=500)


@login_required  
def batch_visualization_data(request, batch_id):
    """JSON API endpoint for batch visualization data (charts, graphs)"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    if batch.status != 'completed':
        return JsonResponse({
            'error': 'Batch analysis not completed',
            'status': batch.status
        }, status=400)
    
    try:
        batch_stats = calculate_batch_statistics(batch)
        if not batch_stats:
            return JsonResponse({
                'error': 'No data available for visualization',
                'status': batch.status
            }, status=404)
        
        # Prepare visualization-specific data structures
        visualization_data = {
            'boxplot_data': _prepare_boxplot_data(batch_stats),
            'histogram_data': _prepare_histogram_data(batch),
            'correlation_data': _prepare_correlation_data(batch_stats),
            'scatter_data': _prepare_scatter_data(batch),
            'summary_metrics': {
                'total_cells': batch_stats['total_cells_across_all_images'],
                'total_images': batch_stats['total_images_analyzed'],
                'avg_cells_per_image': batch_stats['average_cells_per_image'],
                'scale_available': batch_stats['scale_available']
            }
        }
        
        return JsonResponse({
            'success': True,
            'batch_id': batch.id,
            'batch_name': batch.name,
            'visualization_data': visualization_data
        })
        
    except Exception as e:
        logger.error(f"Error generating batch visualization data for batch {batch_id}: {str(e)}")
        return JsonResponse({
            'error': f'Failed to generate visualization data: {str(e)}'
        }, status=500)


@login_required
def batch_feature_comparison_json(request, batch_id):
    """JSON API endpoint for cross-analysis feature comparison data"""
    batch = get_object_or_404(AnalysisBatch, id=batch_id, user=request.user)
    
    if batch.status != 'completed':
        return JsonResponse({
            'error': 'Batch analysis not completed',
            'status': batch.status
        }, status=400)
    
    try:
        batch_stats = calculate_batch_statistics(batch)
        if not batch_stats or 'analysis_comparison' not in batch_stats:
            return JsonResponse({
                'error': 'No comparison data available',
                'status': batch.status
            }, status=404)
        
        comparison_data = batch_stats['analysis_comparison']
        
        # Format data for frontend consumption
        formatted_data = {
            'individual_analyses': comparison_data.get('individual_analyses', []),
            'cross_analysis_metrics': comparison_data.get('cross_analysis_metrics', {}),
            'consistency_scores': _calculate_consistency_scores(comparison_data),
            'quality_indicators': _calculate_quality_indicators(comparison_data)
        }
        
        return JsonResponse({
            'success': True,
            'batch_id': batch.id,
            'batch_name': batch.name,
            'comparison_data': formatted_data
        })
        
    except Exception as e:
        logger.error(f"Error generating batch comparison data for batch {batch_id}: {str(e)}")
        return JsonResponse({
            'error': f'Failed to generate comparison data: {str(e)}'
        }, status=500)


def _prepare_boxplot_data(batch_stats):
    """Prepare data for boxplot visualizations"""
    boxplot_features = [
        'area_stats', 'perimeter_stats', 'circularity_stats', 'eccentricity_stats',
        'solidity_stats', 'extent_stats', 'aspect_ratio_stats',
        'glcm_contrast_stats', 'glcm_correlation_stats', 'glcm_energy_stats',
        'intensity_mean_stats', 'intensity_std_stats', 'intensity_range_stats'
    ]
    
    boxplot_data = {}
    for feature in boxplot_features:
        if feature in batch_stats and batch_stats[feature]:
            stats = batch_stats[feature]
            boxplot_data[feature] = {
                'min': stats.get('min'),
                'q25': stats.get('q25'),
                'median': stats.get('median'),
                'q75': stats.get('q75'),
                'max': stats.get('max'),
                'mean': stats.get('mean'),
                'count': stats.get('count')
            }
    
    return boxplot_data


def _prepare_histogram_data(batch):
    """Prepare histogram data for key features"""
    # Get all detected cells from completed analyses
    completed_analyses = CellAnalysis.objects.filter(
        cell__batch=batch,
        status='completed'
    ).prefetch_related('detected_cells')
    
    histogram_data = {}
    features = ['area', 'circularity', 'eccentricity', 'intensity_mean']
    
    for feature in features:
        values = []
        for analysis in completed_analyses:
            for cell in analysis.detected_cells.all():
                value = getattr(cell, feature, None)
                if value is not None:
                    values.append(float(value))
        
        if values:
            # Create histogram bins
            import numpy as np
            hist, bin_edges = np.histogram(values, bins=20)
            histogram_data[feature] = {
                'values': values,
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'bin_centers': [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            }
    
    return histogram_data


def _prepare_correlation_data(batch_stats):
    """Prepare correlation matrix data"""
    # Select key features for correlation analysis
    correlation_features = [
        ('area_stats', 'Area'),
        ('perimeter_stats', 'Perimeter'),
        ('circularity_stats', 'Circularity'),
        ('eccentricity_stats', 'Eccentricity'),
        ('solidity_stats', 'Solidity'),
        ('aspect_ratio_stats', 'Aspect Ratio'),
        ('intensity_mean_stats', 'Intensity Mean')
    ]
    
    correlation_data = {
        'features': [feature[1] for feature in correlation_features],
        'feature_means': []
    }
    
    for feature_key, feature_name in correlation_features:
        if feature_key in batch_stats and batch_stats[feature_key]:
            correlation_data['feature_means'].append(batch_stats[feature_key].get('mean', 0))
        else:
            correlation_data['feature_means'].append(0)
    
    return correlation_data


def _prepare_scatter_data(batch):
    """Prepare scatter plot data for feature relationships"""
    # Get all detected cells from completed analyses
    completed_analyses = CellAnalysis.objects.filter(
        cell__batch=batch,
        status='completed'
    ).prefetch_related('detected_cells')
    
    scatter_data = {}
    
    # Area vs Circularity
    area_values = []
    circularity_values = []
    
    for analysis in completed_analyses:
        for cell in analysis.detected_cells.all():
            if cell.area is not None and cell.circularity is not None:
                area_values.append(float(cell.area))
                circularity_values.append(float(cell.circularity))
    
    if area_values and circularity_values:
        scatter_data['area_vs_circularity'] = {
            'x': area_values,
            'y': circularity_values,
            'x_label': 'Area (px²)',
            'y_label': 'Circularity'
        }
    
    return scatter_data


def _calculate_consistency_scores(comparison_data):
    """Calculate consistency scores for batch quality assessment"""
    if not comparison_data or 'cross_analysis_metrics' not in comparison_data:
        return {}
    
    metrics = comparison_data['cross_analysis_metrics']
    
    consistency_scores = {}
    
    # Cell count consistency (lower CV is better)
    if 'cell_count_variation' in metrics:
        cv = metrics['cell_count_variation'].get('cv', 0)
        consistency_scores['cell_count_consistency'] = max(0, 100 - cv)
    
    # Area consistency between images
    if 'area_consistency' in metrics:
        cv = metrics['area_consistency'].get('cv_between_images', 0)
        consistency_scores['area_consistency'] = max(0, 100 - cv)
    
    # Circularity consistency between images
    if 'circularity_consistency' in metrics:
        cv = metrics['circularity_consistency'].get('cv_between_images', 0)
        consistency_scores['circularity_consistency'] = max(0, 100 - cv)
    
    # Overall consistency score (average of individual scores)
    if consistency_scores:
        consistency_scores['overall_consistency'] = sum(consistency_scores.values()) / len(consistency_scores)
    
    return consistency_scores


def _calculate_quality_indicators(comparison_data):
    """Calculate quality indicators for the batch analysis"""
    if not comparison_data or 'individual_analyses' not in comparison_data:
        return {}
    
    analyses = comparison_data['individual_analyses']
    
    if not analyses:
        return {}
    
    quality_indicators = {
        'total_analyses': len(analyses),
        'successful_analyses': len([a for a in analyses if a.get('cell_count', 0) > 0]),
        'success_rate': (len([a for a in analyses if a.get('cell_count', 0) > 0]) / len(analyses)) * 100,
        'average_cells_detected': sum(a.get('cell_count', 0) for a in analyses) / len(analyses),
        'processing_efficiency': 'Good' if all(a.get('processing_time', 0) < 300 for a in analyses) else 'Moderate'
    }
    
    return quality_indicators
