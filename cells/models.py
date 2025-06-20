import os
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from PIL import Image


class Cell(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cells')
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='cells/')
    
    # Metadata fields (auto-populated)
    file_size = models.PositiveIntegerField(null=True, blank=True)
    image_width = models.PositiveIntegerField(null=True, blank=True)
    image_height = models.PositiveIntegerField(null=True, blank=True)
    file_format = models.CharField(max_length=10, blank=True)
    
    # Scale calibration
    pixels_per_micron = models.FloatField(null=True, blank=True, help_text="Pixels per micron for scale calibration")
    scale_set = models.BooleanField(default=False, help_text="Whether scale calibration has been set")
    scale_reference_length_pixels = models.FloatField(null=True, blank=True, help_text="Reference length in pixels for calibration")
    scale_reference_length_microns = models.FloatField(null=True, blank=True, help_text="Known real-world length in microns")
    
    # Analysis tracking
    has_analysis = models.BooleanField(default=False)
    analysis_count = models.PositiveIntegerField(default=0)
    
    # Timestamp fields
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
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
    
    def convert_area_to_microns_squared(self, pixel_area):
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
    cellpose_diameter = models.FloatField(default=30.0, help_text="Expected cell diameter in pixels")
    flow_threshold = models.FloatField(default=0.4, help_text="Flow error threshold")
    cellprob_threshold = models.FloatField(default=0.0, help_text="Cell probability threshold")
    
    # ROI (Region of Interest) selection
    use_roi = models.BooleanField(default=False, help_text="Whether to use ROI selection")
    roi_regions = models.JSONField(default=list, blank=True, help_text="ROI regions as list of rectangles [x, y, width, height]")
    roi_count = models.PositiveIntegerField(default=0, help_text="Number of ROI regions selected")
    
    # Image preprocessing options
    apply_preprocessing = models.BooleanField(default=False, help_text="Whether to apply image preprocessing")
    preprocessing_options = models.JSONField(default=dict, blank=True, help_text="Preprocessing configuration options")
    preprocessing_applied = models.BooleanField(default=False, help_text="Whether preprocessing was actually applied")
    preprocessing_steps = models.JSONField(default=list, blank=True, help_text="List of preprocessing steps that were applied")
    
    # Image quality assessment
    quality_metrics = models.JSONField(default=dict, blank=True, help_text="Image quality assessment metrics")
    quality_score = models.FloatField(null=True, blank=True, help_text="Overall image quality score (0-100)")
    quality_category = models.CharField(max_length=20, blank=True, help_text="Quality category: excellent, good, fair, poor")
    
    # Results
    num_cells_detected = models.PositiveIntegerField(default=0)
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)
    
    # Timestamps
    analysis_date = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-analysis_date']
        verbose_name_plural = 'Cell Analyses'
    
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
            return "No preprocessing applied"
        
        return "; ".join(self.preprocessing_steps)
    
    def get_quality_summary(self):
        """Get human-readable summary of image quality"""
        if not self.quality_metrics:
            return "Quality not assessed"
        
        score = self.quality_score or 0
        category = self.quality_category or 'unknown'
        
        return f"{category.title()} quality (score: {score:.1f}/100)"


class DetectedCell(models.Model):
    analysis = models.ForeignKey(CellAnalysis, on_delete=models.CASCADE, related_name='detected_cells')
    cell_id = models.PositiveIntegerField(help_text="Cellpose assigned cell ID")
    
    # Basic measurements (pixels)
    area = models.FloatField(help_text="Area in pixels²")
    perimeter = models.FloatField(help_text="Perimeter in pixels")
    
    # Physical measurements (microns) - calculated if scale is set
    area_microns_sq = models.FloatField(null=True, blank=True, help_text="Area in μm²")
    perimeter_microns = models.FloatField(null=True, blank=True, help_text="Perimeter in μm")
    
    # Shape descriptors
    circularity = models.FloatField(help_text="4π×area/perimeter²")
    eccentricity = models.FloatField(help_text="Eccentricity of the fitted ellipse")
    solidity = models.FloatField(help_text="Area/convex_area ratio")
    extent = models.FloatField(help_text="Area/bounding_box_area ratio")
    
    # Ellipse fitting (pixels)
    major_axis_length = models.FloatField(help_text="Major axis length in pixels")
    minor_axis_length = models.FloatField(help_text="Minor axis length in pixels")
    aspect_ratio = models.FloatField(help_text="Major/minor axis ratio")
    
    # Ellipse fitting (microns) - calculated if scale is set
    major_axis_length_microns = models.FloatField(null=True, blank=True, help_text="Major axis length in μm")
    minor_axis_length_microns = models.FloatField(null=True, blank=True, help_text="Minor axis length in μm")
    
    # Position
    centroid_x = models.FloatField(help_text="X coordinate of centroid")
    centroid_y = models.FloatField(help_text="Y coordinate of centroid")
    
    # Bounding box
    bounding_box_x = models.PositiveIntegerField(help_text="X coordinate of bounding box")
    bounding_box_y = models.PositiveIntegerField(help_text="Y coordinate of bounding box")
    bounding_box_width = models.PositiveIntegerField(help_text="Width of bounding box")
    bounding_box_height = models.PositiveIntegerField(help_text="Height of bounding box")
    
    class Meta:
        ordering = ['cell_id']
        unique_together = ['analysis', 'cell_id']
    
    def __str__(self):
        return f"Cell {self.cell_id} from {self.analysis}"
