from django.contrib import messages
from django.utils.translation import gettext as _
from django.core.paginator import Paginator
import json

from .models import Cell, CellAnalysis, DetectedCell, AnalysisBatch
from .analysis import run_cell_analysis, get_analysis_summary


class CellUploadService:
    @staticmethod
    def create_cell(form, user):
        """Create a new cell with the given form data and user"""
        cell = form.save(commit=False)
        cell.user = user
        cell.save()
        return cell


class CellAnalysisService:
    @staticmethod
    def create_analysis(form, cell, request):
        """Create and process a new cell analysis"""
        analysis = form.save(commit=False)
        analysis.cell = cell
        
        # Handle ROI data if provided
        if analysis.use_roi and 'roi_data' in request.POST:
            try:
                roi_data_str = request.POST.get('roi_data', '[]')
                roi_data = json.loads(roi_data_str)
                roi_count = int(request.POST.get('roi_count', '0'))
                
                analysis.roi_regions = roi_data
                analysis.roi_count = roi_count
                
                messages.success(request, _('ROI selection enabled with {} region(s)').format(roi_count))
            except (json.JSONDecodeError, ValueError):
                messages.warning(request, _('ROI data was invalid, proceeding without ROI selection'))
                analysis.use_roi = False
                analysis.roi_regions = []
                analysis.roi_count = 0
        elif analysis.use_roi:
            messages.warning(request, _('ROI selection was enabled but no regions were drawn. Running standard analysis.'))
            analysis.use_roi = False
        
        analysis.save()
        return analysis
    
    @staticmethod
    def run_analysis(analysis_id):
        """Run the cell analysis process"""
        return run_cell_analysis(analysis_id)
    
    @staticmethod
    def get_analysis_context(analysis, request):
        """Get context data for analysis detail view"""
        context = {
            'analysis': analysis,
            'cell': analysis.cell,
            'summary': get_analysis_summary(analysis) if analysis.status == 'completed' else None,
        }
        
        if analysis.status == 'completed':
            # Paginate detected cells
            detected_cells = analysis.detected_cells.all()
            paginator = Paginator(detected_cells, 20)
            page_number = request.GET.get('page')
            page_obj = paginator.get_page(page_number)
            context['page_obj'] = page_obj
            
            # Add cell filtering information
            validated_cell_count = detected_cells.count()
            context['validated_cell_count'] = validated_cell_count
            
            original_cell_count = analysis.num_cells_detected if analysis.num_cells_detected > 0 else validated_cell_count
            context['original_cell_count'] = original_cell_count
            context['cells_filtered'] = max(0, original_cell_count - validated_cell_count)
            
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
        
        return context
    
    @staticmethod
    def get_analysis_list_context(user, request):
        """Get context for analysis list view"""
        analyses = CellAnalysis.objects.filter(cell__user=user).order_by('-analysis_date')
        
        # Filter by status if provided
        status_filter = request.GET.get('status')
        if status_filter:
            analyses = analyses.filter(status=status_filter)
        
        # Paginate results
        paginator = Paginator(analyses, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        return {
            'page_obj': page_obj,
            'status_filter': status_filter,
            'status_choices': CellAnalysis.STATUS_CHOICES,
        }


class CellExportService:
    @staticmethod
    def generate_csv_response(analysis):
        """Generate CSV export response for analysis"""
        from django.http import HttpResponse
        import csv
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="analysis_{analysis.id}_cells.csv"'
        
        writer = csv.writer(response)
        
        # Write header
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
            
            # Add GLCM texture features
            row.extend([
                cell.glcm_contrast or '', cell.glcm_correlation or '', cell.glcm_energy or '',
                cell.glcm_homogeneity or '', cell.glcm_entropy or '', cell.glcm_variance or '',
                cell.glcm_sum_average or '', cell.glcm_sum_variance or '', cell.glcm_sum_entropy or '',
                cell.glcm_difference_average or '', cell.glcm_difference_variance or '', 
                cell.glcm_difference_entropy or ''
            ])
            
            # Add first-order statistical features
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


class CellScaleService:
    @staticmethod
    def set_scale_calibration(cell, pixels, microns):
        """Set scale calibration for a cell"""
        cell.set_scale_calibration(pixels, microns)
        return cell


class BatchProcessingService:
    """
    Conservative batch processing service optimized for RTX 2070 Max-Q
    Sequential processing to avoid GPU memory overflow
    """
    
    @staticmethod
    def start_batch_processing(batch_id):
        """Start processing a batch of images sequentially"""
        import logging
        import time
        from django.utils import timezone
        
        logger = logging.getLogger(__name__)
        
        try:
            batch = AnalysisBatch.objects.get(id=batch_id)
            
            if batch.status != 'pending':
                logger.warning(f"Batch {batch_id} is not in pending status: {batch.status}")
                return False
            
            # Mark batch as started
            batch.start_processing()
            
            # Get all cells in the batch
            batch_cells = batch.cells.all().order_by('id')
            
            logger.info(f"Starting batch processing for {batch.name} with {len(batch_cells)} images")
            
            processed_count = 0
            failed_count = 0
            start_time = time.time()
            
            # Process each cell sequentially (RTX 2070 Max-Q optimization)
            for cell in batch_cells:
                try:
                    # Create analysis with batch parameters
                    analysis = BatchProcessingService._create_analysis_from_batch(cell, batch)
                    
                    # Run analysis for this cell
                    success = CellAnalysisService.run_analysis(analysis.id)
                    
                    if success:
                        processed_count += 1
                        logger.info(f"Processed cell {cell.id} successfully ({processed_count}/{len(batch_cells)})")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process cell {cell.id} ({failed_count} failures)")
                    
                    # Update batch progress
                    batch.processed_images = processed_count
                    batch.failed_images = failed_count
                    batch.save()
                    
                    # Force garbage collection and GPU memory cleanup
                    import gc
                    gc.collect()
                    
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    
                    # Small delay to prevent overheating (RTX 2070 Max-Q consideration)
                    time.sleep(1)
                    
                except Exception as e:
                    failed_count += 1
                    batch.failed_images = failed_count
                    batch.save()
                    logger.error(f"Error processing cell {cell.id}: {str(e)}")
                    continue
            
            # Calculate final statistics and complete batch
            total_time = time.time() - start_time
            batch.batch_processing_time = total_time
            
            # Calculate total cells detected
            total_cells = 0
            completed_analyses = CellAnalysis.objects.filter(
                cell__batch=batch,
                status='completed'
            )
            for analysis in completed_analyses:
                total_cells += analysis.num_cells_detected
            
            batch.total_cells_detected = total_cells
            batch.average_cells_per_image = total_cells / processed_count if processed_count > 0 else 0
            
            # Mark batch as completed or failed
            if processed_count > 0:
                batch.complete_processing()
                logger.info(f"Batch {batch.name} completed successfully: {processed_count} processed, {failed_count} failed")
            else:
                batch.fail_processing("All images failed to process")
                logger.error(f"Batch {batch.name} failed: no images were processed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Batch processing failed for batch {batch_id}: {str(e)}")
            try:
                batch = AnalysisBatch.objects.get(id=batch_id)
                batch.fail_processing(str(e))
            except:
                pass
            return False
    
    @staticmethod
    def _create_analysis_from_batch(cell, batch):
        """Create a CellAnalysis object from batch parameters"""
        params = batch.analysis_parameters
        
        # Create analysis with shared parameters
        analysis = CellAnalysis.objects.create(
            cell=cell,
            cellpose_model=params.get('cellpose_model', 'cpsam'),
            cellpose_diameter=params.get('cellpose_diameter', 0.0),
            flow_threshold=params.get('flow_threshold', 0.4),
            cellprob_threshold=params.get('cellprob_threshold', 0.0),
            filtering_mode=params.get('filtering_mode', 'clinical'),
            apply_preprocessing=params.get('apply_preprocessing', False),
            preprocessing_options=params.get('preprocessing_options', {}),
            use_roi=False,  # ROI disabled for batch processing
        )
        
        # Apply filtering preset
        analysis.apply_filtering_preset(analysis.filtering_mode)
        analysis.save()
        
        return analysis
    
    @staticmethod
    def cancel_batch_processing(batch_id):
        """Cancel batch processing (simplified - just mark as cancelled)"""
        try:
            batch = AnalysisBatch.objects.get(id=batch_id)
            if batch.status == 'processing':
                batch.status = 'cancelled'
                batch.save()
                return True
        except AnalysisBatch.DoesNotExist:
            pass
        return False
    
    @staticmethod
    def get_batch_progress(batch_id):
        """Get current batch processing progress"""
        try:
            batch = AnalysisBatch.objects.get(id=batch_id)
            return {
                'status': batch.status,
                'progress_percentage': batch.progress_percentage,
                'processed_images': batch.processed_images,
                'total_images': batch.total_images,
                'failed_images': batch.failed_images,
            }
        except AnalysisBatch.DoesNotExist:
            return None