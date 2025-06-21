import os
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from PIL import Image


class Cell(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cells')
    name = models.CharField(max_length=255, verbose_name=_('Name'))
    image = models.ImageField(upload_to='cells/', verbose_name=_('Image'))
    
    # Metadata fields (auto-populated)
    file_size = models.PositiveIntegerField(null=True, blank=True, verbose_name=_('File Size'))
    image_width = models.PositiveIntegerField(null=True, blank=True, verbose_name=_('Image Width'))
    image_height = models.PositiveIntegerField(null=True, blank=True, verbose_name=_('Image Height'))
    file_format = models.CharField(max_length=10, blank=True, verbose_name=_('File Format'))
    
    # Scale calibration
    pixels_per_micron = models.FloatField(null=True, blank=True, verbose_name=_('Pixels per Micron'), help_text=_('Pixels per micron for scale calibration'))
    scale_set = models.BooleanField(default=False, verbose_name=_('Scale Set'), help_text=_('Whether scale calibration has been set'))
    scale_reference_length_pixels = models.FloatField(null=True, blank=True, verbose_name=_('Reference Length (pixels)'), help_text=_('Reference length in pixels for calibration'))
    scale_reference_length_microns = models.FloatField(null=True, blank=True, verbose_name=_('Reference Length (microns)'), help_text=_('Known real-world length in microns'))
    
    # Analysis tracking
    has_analysis = models.BooleanField(default=False, verbose_name=_('Has Analysis'))
    analysis_count = models.PositiveIntegerField(default=0, verbose_name=_('Analysis Count'))
    
    # Timestamp fields
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_('Created At'))
    modified_at = models.DateTimeField(auto_now=True, verbose_name=_('Modified At'))
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.user.email}"
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        
        # Extract and save metadata after the image is saved
        if self.image:
            self._extract_metadata()
    
    def _extract_metadata(self):
        try:
            # Get file size
            self.file_size = self.image.size
            
            # Open image with PIL to get dimensions and format
            with Image.open(self.image.path) as img:
                self.image_width = img.width
                self.image_height = img.height
                self.file_format = img.format.lower() if img.format else ''
            
            # Save without triggering save recursion
            Cell.objects.filter(pk=self.pk).update(
                file_size=self.file_size,
                image_width=self.image_width,
                image_height=self.image_height,
                file_format=self.file_format
            )
        except Exception as e:
            # Handle any errors in metadata extraction gracefully
            pass
    
    @property
    def latest_analysis(self):
        return self.analyses.filter(status='completed').order_by('-analysis_date').first()
    
    def set_scale_calibration(self, pixels, microns):
        """Set scale calibration from reference measurements"""
        if pixels > 0 and microns > 0:
            self.scale_reference_length_pixels = pixels
            self.scale_reference_length_microns = microns
            self.pixels_per_micron = pixels / microns
            self.scale_set = True
            self.save()
    
    def convert_pixels_to_microns(self, pixels):
        """Convert pixel measurement to microns"""
        if self.scale_set and self.pixels_per_micron:
            return pixels / self.pixels_per_micron
        return None


        """Convert pixel area to square microns"""
        if self.scale_set and self.pixels_per_micron:
            return pixel_area / (self.pixels_per_micron ** 2)
        return None


class CellAnalysis(models.Model):
    STATUS_CHOICES = [
        ('pending', _('Pending')),
        ('processing', _('Processing')),
        ('completed', _('Completed')),
        ('failed', _('Failed')),
    ]
    
    CELLPOSE_MODEL_CHOICES = [
        ('cyto', _('Cytoplasm')),
        ('nuclei', _('Nuclei')),
        ('cyto2', _('Cytoplasm 2.0')),
        ('custom', _('Custom')),
    ]
    
    cell = models.ForeignKey(Cell, on_delete=models.CASCADE, related_name='analyses')
    segmentation_image = models.ImageField(upload_to='analyses/segmentation/', null=True, blank=True)
    
    # Analysis parameters
    cellpose_model = models.CharField(max_length=20, choices=CELLPOSE_MODEL_CHOICES, default='cyto')
    cellpose_diameter = models.FloatField(default=30.0, verbose_name=_('Cell Diameter'), help_text=_('Expected cell diameter in pixels'))
    flow_threshold = models.FloatField(default=0.4, verbose_name=_('Flow Threshold'), help_text=_('Flow error threshold'))
    cellprob_threshold = models.FloatField(default=0.0, verbose_name=_('Cell Probability Threshold'), help_text=_('Cell probability threshold'))
    
    # ROI (Region of Interest) selection
    use_roi = models.BooleanField(default=False, verbose_name=_('Use ROI'), help_text=_('Whether to use ROI selection'))
    roi_regions = models.JSONField(default=list, blank=True, verbose_name=_('ROI Regions'), help_text=_('ROI regions as list of rectangles [x, y, width, height]'))
    roi_count = models.PositiveIntegerField(default=0, verbose_name=_('ROI Count'), help_text=_('Number of ROI regions selected'))
    
    # Image preprocessing options
    apply_preprocessing = models.BooleanField(default=False, verbose_name=_('Apply Preprocessing'), help_text=_('Whether to apply image preprocessing'))
    preprocessing_options = models.JSONField(default=dict, blank=True, verbose_name=_('Preprocessing Options'), help_text=_('Preprocessing configuration options'))
    preprocessing_applied = models.BooleanField(default=False, verbose_name=_('Preprocessing Applied'), help_text=_('Whether preprocessing was actually applied'))
    preprocessing_steps = models.JSONField(default=list, blank=True, verbose_name=_('Preprocessing Steps'), help_text=_('List of preprocessing steps that were applied'))
    
    # Image quality assessment
    quality_metrics = models.JSONField(default=dict, blank=True, verbose_name=_('Quality Metrics'), help_text=_('Image quality assessment metrics'))
    quality_score = models.FloatField(null=True, blank=True, verbose_name=_('Quality Score'), help_text=_('Overall image quality score (0-100)'))
    quality_category = models.CharField(max_length=20, blank=True, verbose_name=_('Quality Category'), help_text=_('Quality category: excellent, good, fair, poor'))
    
    # Cell filtering configuration
    FILTERING_MODE_CHOICES = [
        ('none', _('No Filtering')),
        ('basic', _('Basic Filtering')),
        ('research', _('Research Mode')),
        ('clinical', _('Clinical Mode')),
        ('custom', _('Custom Settings')),
    ]
    
    filtering_mode = models.CharField(
        max_length=20, 
        choices=FILTERING_MODE_CHOICES, 
        default='clinical',
        verbose_name=_('Filtering Mode'), 
        help_text=_('Cell filtering strictness level')
    )
    
    # Segmentation refinement options
    enable_size_filtering = models.BooleanField(default=True, verbose_name=_('Size Filtering'), help_text=_('Remove cells outside size range'))
    min_cell_area = models.FloatField(default=50, verbose_name=_('Min Cell Area'), help_text=_('Minimum cell area in pixels'))
    max_cell_area = models.FloatField(null=True, blank=True, verbose_name=_('Max Cell Area'), help_text=_('Maximum cell area in pixels (blank = no limit)'))
    
    enable_shape_filtering = models.BooleanField(default=True, verbose_name=_('Shape Filtering'), help_text=_('Remove non-cellular shapes'))
    min_circularity = models.FloatField(default=0.1, verbose_name=_('Min Circularity'), help_text=_('Minimum circularity (0-1)'))
    max_eccentricity = models.FloatField(default=0.95, verbose_name=_('Max Eccentricity'), help_text=_('Maximum eccentricity (0-1)'))
    min_solidity = models.FloatField(default=0.7, verbose_name=_('Min Solidity'), help_text=_('Minimum solidity (0-1)'))
    
    enable_edge_removal = models.BooleanField(default=False, verbose_name=_('Edge Removal'), help_text=_('Remove cells touching image edges'))
    edge_border_width = models.IntegerField(default=5, verbose_name=_('Edge Border Width'), help_text=_('Border width for edge removal'))
    
    enable_watershed = models.BooleanField(default=False, verbose_name=_('Watershed Splitting'), help_text=_('Split touching cells using watershed'))
    watershed_min_distance = models.IntegerField(default=10, verbose_name=_('Watershed Distance'), help_text=_('Minimum distance for watershed peaks'))
    
    # Morphometric validation options
    enable_outlier_removal = models.BooleanField(default=True, verbose_name=_('Outlier Removal'), help_text=_('Remove statistical outliers'))
    outlier_method = models.CharField(
        max_length=20,
        choices=[
            ('iqr', _('IQR Method')),
            ('zscore', _('Z-Score Method')),
            ('modified_zscore', _('Modified Z-Score')),
        ],
        default='iqr',
        verbose_name=_('Outlier Method'),
        help_text=_('Statistical method for outlier detection')
    )
    outlier_threshold = models.FloatField(default=1.5, verbose_name=_('Outlier Threshold'), help_text=_('Threshold for outlier detection'))
    
    enable_physics_validation = models.BooleanField(default=True, verbose_name=_('Physics Validation'), help_text=_('Remove cells violating physical constraints'))
    
    # Results
    num_cells_detected = models.PositiveIntegerField(default=0, verbose_name=_('Cells Detected'))
    processing_time = models.FloatField(null=True, blank=True, verbose_name=_('Processing Time'), help_text=_('Processing time in seconds'))
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name=_('Status'))
    error_message = models.TextField(blank=True, verbose_name=_('Error Message'))
    
    # Timestamps
    analysis_date = models.DateTimeField(auto_now_add=True, verbose_name=_('Analysis Date'))
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name=_('Completed At'))
    
    class Meta:
        ordering = ['-analysis_date']
        verbose_name_plural = _('Cell Analyses')
    
    def __str__(self):
        status_info = self.status
        if self.apply_preprocessing:
            status_info += " (with preprocessing)"
        return f"Analysis of {self.cell.name} - {status_info}"
    
    def save(self, *args, **kwargs):
        # Extract quality score from quality_metrics if available
        if self.quality_metrics and 'overall_score' in self.quality_metrics:
            self.quality_score = self.quality_metrics['overall_score']
            self.quality_category = self.quality_metrics.get('quality_category', '')
        
        super().save(*args, **kwargs)
        
        # Update parent cell analysis tracking
        self.cell.analysis_count = self.cell.analyses.count()
        self.cell.has_analysis = self.cell.analyses.filter(status='completed').exists()
        Cell.objects.filter(pk=self.cell.pk).update(
            has_analysis=self.cell.has_analysis,
            analysis_count=self.cell.analysis_count
        )
    
    def get_preprocessing_summary(self):
        """Get human-readable summary of preprocessing steps"""
        if not self.preprocessing_applied or not self.preprocessing_steps:
            return _('No preprocessing applied')
        
        return "; ".join(self.preprocessing_steps)
    
    def get_quality_summary(self):
        """Get human-readable summary of image quality"""
        if not self.quality_metrics:
            return _('Quality not assessed')
        
        score = self.quality_score or 0
        category = self.quality_category or 'unknown'
        
        return f"{category.title()} quality (score: {score:.1f}/100)"
    
    def apply_filtering_preset(self, mode):
        """Apply predefined filtering settings based on mode"""
        if mode == 'none':
            # No filtering - preserve all detected cells
            self.enable_size_filtering = False
            self.enable_shape_filtering = False
            self.enable_edge_removal = False
            self.enable_watershed = False
            self.enable_outlier_removal = False
            self.enable_physics_validation = False
            
        elif mode == 'basic':
            # Minimal filtering - only obvious artifacts
            self.enable_size_filtering = True
            self.min_cell_area = 25  # Very permissive
            self.max_cell_area = None
            self.enable_shape_filtering = False
            self.enable_edge_removal = False
            self.enable_watershed = False
            self.enable_outlier_removal = False
            self.enable_physics_validation = True  # Keep physics checks
            
        elif mode == 'research':
            # Conservative filtering for research applications
            self.enable_size_filtering = True
            self.min_cell_area = 40
            self.max_cell_area = None
            self.enable_shape_filtering = True
            self.min_circularity = 0.05  # Very permissive
            self.max_eccentricity = 0.98
            self.min_solidity = 0.5
            self.enable_edge_removal = False
            self.enable_watershed = False
            self.enable_outlier_removal = True
            self.outlier_method = 'modified_zscore'
            self.outlier_threshold = 2.5  # Less strict
            self.enable_physics_validation = True
            
        elif mode == 'clinical':
            # Standard clinical filtering (current default)
            self.enable_size_filtering = True
            self.min_cell_area = 50
            self.max_cell_area = None
            self.enable_shape_filtering = True
            self.min_circularity = 0.1
            self.max_eccentricity = 0.95
            self.min_solidity = 0.7
            self.enable_edge_removal = False
            self.enable_watershed = False
            self.enable_outlier_removal = True
            self.outlier_method = 'iqr'
            self.outlier_threshold = 1.5
            self.enable_physics_validation = True
            
        # For 'custom' mode, don't change settings - user controls them
        
        # Update the filtering mode
        self.filtering_mode = mode
    
    def get_filtering_options(self):
        """Get current filtering options as a dictionary for the analysis pipeline"""
        return {
            # Segmentation refinement options
            'apply_size_filtering': self.enable_size_filtering,
            'min_cell_area': self.min_cell_area,
            'max_cell_area': self.max_cell_area,
            'apply_shape_filtering': self.enable_shape_filtering,
            'min_circularity': self.min_circularity,
            'max_eccentricity': self.max_eccentricity,
            'min_solidity': self.min_solidity,
            'apply_watershed': self.enable_watershed,
            'watershed_min_distance': self.watershed_min_distance,
            'apply_smoothing': True,  # Always apply boundary smoothing
            'smoothing_factor': 1.0,
            'remove_edge_cells': self.enable_edge_removal,
            'border_width': self.edge_border_width,
            
            # Morphometric validation options
            'enable_outlier_removal': self.enable_outlier_removal,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'enable_physics_validation': self.enable_physics_validation,
        }


class DetectedCell(models.Model):
    analysis = models.ForeignKey(CellAnalysis, on_delete=models.CASCADE, related_name='detected_cells')
    cell_id = models.PositiveIntegerField(verbose_name=_('Cell ID'), help_text=_('Cellpose assigned cell ID'))
    
    # Basic measurements (pixels)
    area = models.FloatField(verbose_name=_('Area'), help_text=_('Area in pixels²'))
    perimeter = models.FloatField(verbose_name=_('Perimeter'), help_text=_('Perimeter in pixels'))
    
    # Physical measurements (microns) - calculated if scale is set
    area_microns_sq = models.FloatField(null=True, blank=True, verbose_name=_('Area (μm²)'), help_text=_('Area in μm²'))
    perimeter_microns = models.FloatField(null=True, blank=True, verbose_name=_('Perimeter (μm)'), help_text=_('Perimeter in μm'))
    
    # Shape descriptors
    circularity = models.FloatField(verbose_name=_('Circularity'), help_text=_('4π×area/perimeter²'))
    eccentricity = models.FloatField(verbose_name=_('Eccentricity'), help_text=_('Eccentricity of the fitted ellipse'))
    solidity = models.FloatField(verbose_name=_('Solidity'), help_text=_('Area/convex_area ratio'))
    extent = models.FloatField(verbose_name=_('Extent'), help_text=_('Area/bounding_box_area ratio'))
    
    # Ellipse fitting (pixels)
    major_axis_length = models.FloatField(verbose_name=_('Major Axis Length'), help_text=_('Major axis length in pixels'))
    minor_axis_length = models.FloatField(verbose_name=_('Minor Axis Length'), help_text=_('Minor axis length in pixels'))
    aspect_ratio = models.FloatField(verbose_name=_('Aspect Ratio'), help_text=_('Major/minor axis ratio'))
    
    # Ellipse fitting (microns) - calculated if scale is set
    major_axis_length_microns = models.FloatField(null=True, blank=True, verbose_name=_('Major Axis Length (μm)'), help_text=_('Major axis length in μm'))
    minor_axis_length_microns = models.FloatField(null=True, blank=True, verbose_name=_('Minor Axis Length (μm)'), help_text=_('Minor axis length in μm'))
    
    # Position
    centroid_x = models.FloatField(verbose_name=_('Centroid X'), help_text=_('X coordinate of centroid'))
    centroid_y = models.FloatField(verbose_name=_('Centroid Y'), help_text=_('Y coordinate of centroid'))
    
    # Bounding box
    bounding_box_x = models.PositiveIntegerField(verbose_name=_('Bounding Box X'), help_text=_('X coordinate of bounding box'))
    bounding_box_y = models.PositiveIntegerField(verbose_name=_('Bounding Box Y'), help_text=_('Y coordinate of bounding box'))
    bounding_box_width = models.PositiveIntegerField(verbose_name=_('Bounding Box Width'), help_text=_('Width of bounding box'))
    bounding_box_height = models.PositiveIntegerField(verbose_name=_('Bounding Box Height'), help_text=_('Height of bounding box'))
    
    # GLCM Texture Features
    glcm_contrast = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Contrast'), help_text=_('Measure of local intensity variation'))
    glcm_correlation = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Correlation'), help_text=_('Measure of linear dependency of gray levels'))
    glcm_energy = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Energy'), help_text=_('Measure of textural uniformity'))
    glcm_homogeneity = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Homogeneity'), help_text=_('Measure of closeness of distribution'))
    glcm_entropy = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Entropy'), help_text=_('Measure of randomness'))
    glcm_variance = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Variance'), help_text=_('Measure of heterogeneity'))
    glcm_sum_average = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Sum Average'), help_text=_('Sum average of GLCM'))
    glcm_sum_variance = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Sum Variance'), help_text=_('Sum variance of GLCM'))
    glcm_sum_entropy = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Sum Entropy'), help_text=_('Sum entropy of GLCM'))
    glcm_difference_average = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Difference Average'), help_text=_('Difference average of GLCM'))
    glcm_difference_variance = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Difference Variance'), help_text=_('Difference variance of GLCM'))
    glcm_difference_entropy = models.FloatField(null=True, blank=True, verbose_name=_('GLCM Difference Entropy'), help_text=_('Difference entropy of GLCM'))
    
    # First-Order Statistical Features
    intensity_mean = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Mean'), help_text=_('Mean intensity value'))
    intensity_std = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Std'), help_text=_('Standard deviation of intensity'))
    intensity_variance = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Variance'), help_text=_('Variance of intensity'))
    intensity_skewness = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Skewness'), help_text=_('Skewness of intensity distribution'))
    intensity_kurtosis = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Kurtosis'), help_text=_('Kurtosis of intensity distribution'))
    intensity_min = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Min'), help_text=_('Minimum intensity value'))
    intensity_max = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Max'), help_text=_('Maximum intensity value'))
    intensity_range = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Range'), help_text=_('Range of intensity values'))
    intensity_p10 = models.FloatField(null=True, blank=True, verbose_name=_('Intensity 10th Percentile'), help_text=_('10th percentile of intensity'))
    intensity_p25 = models.FloatField(null=True, blank=True, verbose_name=_('Intensity 25th Percentile'), help_text=_('25th percentile of intensity'))
    intensity_p75 = models.FloatField(null=True, blank=True, verbose_name=_('Intensity 75th Percentile'), help_text=_('75th percentile of intensity'))
    intensity_p90 = models.FloatField(null=True, blank=True, verbose_name=_('Intensity 90th Percentile'), help_text=_('90th percentile of intensity'))
    intensity_iqr = models.FloatField(null=True, blank=True, verbose_name=_('Intensity IQR'), help_text=_('Interquartile range of intensity'))
    intensity_entropy = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Entropy'), help_text=_('Entropy of intensity histogram'))
    intensity_energy = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Energy'), help_text=_('Energy of intensity histogram'))
    intensity_median = models.FloatField(null=True, blank=True, verbose_name=_('Intensity Median'), help_text=_('Median intensity value'))
    intensity_mad = models.FloatField(null=True, blank=True, verbose_name=_('Intensity MAD'), help_text=_('Median absolute deviation'))
    intensity_cv = models.FloatField(null=True, blank=True, verbose_name=_('Intensity CV'), help_text=_('Coefficient of variation'))
    
    class Meta:
        ordering = ['cell_id']
        unique_together = ['analysis', 'cell_id']
    
    def __str__(self):
        return f"Cell {self.cell_id} from {self.analysis}"
