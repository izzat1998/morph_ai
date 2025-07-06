import os
from django.db import models
from django.conf import settings
from PIL import Image
from django.utils import timezone

class AnalysisBatch(models.Model):
    """Conservative batch processing model for grouping related cell analyses"""
    
    STATUS_CHOICES = [
        ('pending', 'Ожидание'),
        ('processing', 'Обработка'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
        ('cancelled', 'Отменено'),
    ]
    
    # Basic batch information
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='analysis_batches')
    name = models.CharField(max_length=255, verbose_name='Название пакета')
    description = models.TextField(blank=True, verbose_name='Описание')
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')
    error_message = models.TextField(blank=True, verbose_name='Сообщение об ошибке')
    
    # Shared analysis parameters (JSON field for flexibility)
    analysis_parameters = models.JSONField(default=dict, blank=True, verbose_name='Параметры анализа')
    
    # Progress tracking
    total_images = models.PositiveIntegerField(default=0, verbose_name='Всего изображений')
    processed_images = models.PositiveIntegerField(default=0, verbose_name='Обработано изображений')
    failed_images = models.PositiveIntegerField(default=0, verbose_name='Ошибок обработки')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создано')
    processing_started_at = models.DateTimeField(null=True, blank=True, verbose_name='Обработка начата')
    processing_completed_at = models.DateTimeField(null=True, blank=True, verbose_name='Обработка завершена')
    
    # Simple batch statistics (calculated after processing)
    total_cells_detected = models.PositiveIntegerField(default=0, verbose_name='Всего обнаружено клеток')
    average_cells_per_image = models.FloatField(null=True, blank=True, verbose_name='Среднее количество клеток на изображение')
    batch_processing_time = models.FloatField(null=True, blank=True, verbose_name='Общее время обработки (сек)')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Пакет анализа'
        verbose_name_plural = 'Пакеты анализа'
    
    def __str__(self):
        return f"{self.name} ({self.total_images} images) - {self.status}"
    
    @property
    def progress_percentage(self):
        """Calculate processing progress as percentage"""
        if self.total_images == 0:
            return 0
        return (self.processed_images / self.total_images) * 100
    
    @property
    def processing_duration(self):
        """Calculate total processing duration"""
        if self.processing_started_at and self.processing_completed_at:
            return (self.processing_completed_at - self.processing_started_at).total_seconds()
        return None
    
    def start_processing(self):
        """Mark batch as started processing"""
        self.status = 'processing'
        self.processing_started_at = timezone.now()
        self.save()
    
    def complete_processing(self):
        """Mark batch as completed"""
        self.status = 'completed'
        self.processing_completed_at = timezone.now()
        self.save()
    
    def fail_processing(self, error_message=""):
        """Mark batch as failed"""
        self.status = 'failed'
        self.error_message = error_message
        self.processing_completed_at = timezone.now()
        self.save()


class Cell(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cells')
    name = models.CharField(max_length=255, verbose_name='Название')
    image = models.ImageField(upload_to='cells/', verbose_name='Изображение')
    
    # Optional batch association (nullable to preserve existing data)
    batch = models.ForeignKey(AnalysisBatch, on_delete=models.SET_NULL, null=True, blank=True, 
                             related_name='cells', verbose_name='Пакет анализа')
    
    # Metadata fields (auto-populated)
    file_size = models.PositiveIntegerField(null=True, blank=True, verbose_name='Размер файла')
    image_width = models.PositiveIntegerField(null=True, blank=True, verbose_name='Ширина изображения')
    image_height = models.PositiveIntegerField(null=True, blank=True, verbose_name='Высота изображения')
    file_format = models.CharField(max_length=10, blank=True, verbose_name='Формат файла')
    
    # Scale calibration
    pixels_per_micron = models.FloatField(null=True, blank=True, verbose_name='Пикселей на микрон', help_text='Пикселей на микрон для калибровки масштаба')
    scale_set = models.BooleanField(default=False, verbose_name='Масштаб установлен', help_text='Установлена ли калибровка масштаба')
    scale_reference_length_pixels = models.FloatField(null=True, blank=True, verbose_name='Референсная длина (пиксели)', help_text='Референсная длина в пикселях для калибровки')
    scale_reference_length_microns = models.FloatField(null=True, blank=True, verbose_name='Референсная длина (микроны)', help_text='Известная реальная длина в микронах')
    
    # Analysis tracking
    has_analysis = models.BooleanField(default=False, verbose_name='Есть анализ')
    analysis_count = models.PositiveIntegerField(default=0, verbose_name='Количество анализов')
    
    # Timestamp fields
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создано')
    modified_at = models.DateTimeField(auto_now=True, verbose_name='Изменено')
    
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
        ('pending', 'Ожидание'),
        ('processing', 'Обработка'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
    ]
    
    CELLPOSE_MODEL_CHOICES = [
        ('cyto', 'Цитоплазма'),
        ('nuclei', 'Ядра'),
        ('cyto2', 'Цитоплазма 2.0'),
        ('cpsam', 'CellposeSAM'),
        ("cyto3",'Лучшая моедель'),
        ('custom', 'Пользовательская'),
    ]
    
    cell = models.ForeignKey(Cell, on_delete=models.CASCADE, related_name='analyses')
    
    # Visualization images for different pipeline stages
    segmentation_image = models.ImageField(upload_to='analyses/segmentation/', null=True, blank=True, verbose_name='Визуализация основного конвейера')
    flow_analysis_image = models.ImageField(upload_to='analyses/flow_analysis/', null=True, blank=True, verbose_name='Продвинутый анализ потока')
    style_quality_image = models.ImageField(upload_to='analyses/style_quality/', null=True, blank=True, verbose_name='Анализ стиля и качества')
    edge_boundary_image = models.ImageField(upload_to='analyses/edge_boundary/', null=True, blank=True, verbose_name='Анализ краев и границ')
    
    # Analysis parameters
    cellpose_model = models.CharField(max_length=20, choices=CELLPOSE_MODEL_CHOICES, default='cyto3')
    cellpose_diameter = models.FloatField(default=0.0, verbose_name='Диаметр клетки', help_text='Ожидаемый диаметр клетки в пикселях (0 = автоматическое определение)')
    flow_threshold = models.FloatField(default=0.4, verbose_name='Порог потока', help_text='Порог ошибки потока')
    cellprob_threshold = models.FloatField(default=0.0, verbose_name='Порог вероятности клетки', help_text='Порог вероятности клетки')
    
    # ROI (Region of Interest) selection
    use_roi = models.BooleanField(default=False, verbose_name='Использовать ROI', help_text='Использовать ли выбор области интереса')
    roi_regions = models.JSONField(default=list, blank=True, verbose_name='Области ROI', help_text='Области ROI как список прямоугольников [x, y, ширина, высота]')
    roi_count = models.PositiveIntegerField(default=0, verbose_name='Количество ROI', help_text='Количество выбранных областей ROI')
    
    # Image preprocessing options
    apply_preprocessing = models.BooleanField(default=False, verbose_name='Применить предобработку', help_text='Применять ли предобработку изображения')
    preprocessing_options = models.JSONField(default=dict, blank=True, verbose_name='Опции предобработки', help_text='Опции конфигурации предобработки')
    preprocessing_applied = models.BooleanField(default=False, verbose_name='Предобработка применена', help_text='Была ли фактически применена предобработка')
    preprocessing_steps = models.JSONField(default=list, blank=True, verbose_name='Шаги предобработки', help_text='Список шагов предобработки, которые были применены')
    
    # Image quality assessment
    quality_metrics = models.JSONField(default=dict, blank=True, verbose_name='Метрики качества', help_text='Метрики оценки качества изображения')
    quality_score = models.FloatField(null=True, blank=True, verbose_name='Оценка качества', help_text='Общая оценка качества изображения (0-100)')
    quality_category = models.CharField(max_length=20, blank=True, verbose_name='Категория качества', help_text='Категория качества: отличное, хорошее, удовлетворительное, плохое')
    
    # Cell filtering configuration
    FILTERING_MODE_CHOICES = [
        ('none', 'Без фильтрации'),
        ('basic', 'Базовая фильтрация'),
        ('research', 'Исследовательский режим'),
        ('clinical', 'Клинический режим'),
        ('custom', 'Пользовательские настройки'),
    ]
    
    filtering_mode = models.CharField(
        max_length=20, 
        choices=FILTERING_MODE_CHOICES, 
        default='clinical',
        verbose_name='Режим фильтрации', 
        help_text='Уровень строгости фильтрации клеток'
    )
    
    # Segmentation refinement options
    enable_size_filtering = models.BooleanField(default=True, verbose_name='Фильтрация по размеру', help_text='Удалить клетки вне диапазона размеров')
    min_cell_area = models.FloatField(default=50, verbose_name='Мин. площадь клетки', help_text='Минимальная площадь клетки в пикселях')
    max_cell_area = models.FloatField(null=True, blank=True, verbose_name='Макс. площадь клетки', help_text='Максимальная площадь клетки в пикселях (пусто = без ограничений)')
    
    enable_shape_filtering = models.BooleanField(default=True, verbose_name='Фильтрация по форме', help_text='Удалить некле­точные формы')
    min_circularity = models.FloatField(default=0.1, verbose_name='Мин. круглость', help_text='Минимальная круглость (0-1)')
    max_eccentricity = models.FloatField(default=0.95, verbose_name='Макс. эксцентриситет', help_text='Максимальный эксцентриситет (0-1)')
    min_solidity = models.FloatField(default=0.7, verbose_name='Мин. плотность', help_text='Минимальная плотность (0-1)')
    
    enable_edge_removal = models.BooleanField(default=False, verbose_name='Удаление краевых', help_text='Удалить клетки, касающиеся краев изображения')
    edge_border_width = models.IntegerField(default=5, verbose_name='Ширина краевой границы', help_text='Ширина границы для удаления краевых')
    
    enable_watershed = models.BooleanField(default=False, verbose_name='Разделение водоразделом', help_text='Разделить соприкасающиеся клетки с помощью водораздела')
    watershed_min_distance = models.IntegerField(default=10, verbose_name='Расстояние водораздела', help_text='Минимальное расстояние для пиков водораздела')
    
    # Morphometric validation options
    enable_outlier_removal = models.BooleanField(default=True, verbose_name='Удаление выбросов', help_text='Удалить статистические выбросы')
    outlier_method = models.CharField(
        max_length=20,
        choices=[
            ('iqr', 'Метод IQR'),
            ('zscore', 'Метод Z-оценки'),
            ('modified_zscore', 'Модифицированная Z-оценка'),
        ],
        default='iqr',
        verbose_name='Метод выбросов',
        help_text='Статистический метод для обнаружения выбросов'
    )
    outlier_threshold = models.FloatField(default=1.5, verbose_name='Порог выбросов', help_text='Порог для обнаружения выбросов')
    
    enable_physics_validation = models.BooleanField(default=True, verbose_name='Физическая валидация', help_text='Удалить клетки, нарушающие физические ограничения')
    
    # Results
    num_cells_detected = models.PositiveIntegerField(default=0, verbose_name='Обнаружено клеток')
    processing_time = models.FloatField(null=True, blank=True, verbose_name='Время обработки', help_text='Время обработки в секундах')
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')
    error_message = models.TextField(blank=True, verbose_name='Сообщение об ошибке')
    
    # Timestamps
    analysis_date = models.DateTimeField(auto_now_add=True, verbose_name='Дата анализа')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='Завершено в')
    
    class Meta:
        ordering = ['-analysis_date']
        verbose_name_plural = 'Анализы клеток'
    
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
            return 'Предобработка не применялась'
        
        return "; ".join(self.preprocessing_steps)
    
    def get_quality_summary(self):
        """Get human-readable summary of image quality"""
        if not self.quality_metrics:
            return 'Качество не оценивалось'
        
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
    cell_id = models.PositiveIntegerField(verbose_name='ID клетки', help_text='ID клетки, назначенный Cellpose')
    
    # Basic measurements (pixels)
    area = models.FloatField(verbose_name='Площадь', help_text='Площадь в пикселях²')
    perimeter = models.FloatField(verbose_name='Периметр', help_text='Периметр в пикселях')
    
    # Physical measurements (microns) - calculated if scale is set
    area_microns_sq = models.FloatField(null=True, blank=True, verbose_name='Площадь (мкм²)', help_text='Площадь в мкм²')
    perimeter_microns = models.FloatField(null=True, blank=True, verbose_name='Периметр (мкм)', help_text='Периметр в мкм')
    
    # Shape descriptors
    circularity = models.FloatField(verbose_name='Круглость', help_text='4π×площадь/периметр²')
    eccentricity = models.FloatField(verbose_name='Эксцентриситет', help_text='Эксцентриситет аппроксимирующего эллипса')
    solidity = models.FloatField(verbose_name='Плотность', help_text='Отношение площадь/выпуклая_площадь')
    extent = models.FloatField(verbose_name='Протяженность', help_text='Отношение площадь/площадь_охватывающего_прямоугольника')
    
    # Ellipse fitting (pixels)
    major_axis_length = models.FloatField(verbose_name='Длина большой оси', help_text='Длина большой оси в пикселях')
    minor_axis_length = models.FloatField(verbose_name='Длина малой оси', help_text='Длина малой оси в пикселях')
    aspect_ratio = models.FloatField(verbose_name='Соотношение сторон', help_text='Отношение большой/малой оси')
    
    # Ellipse fitting (microns) - calculated if scale is set
    major_axis_length_microns = models.FloatField(null=True, blank=True, verbose_name='Длина большой оси (мкм)', help_text='Длина большой оси в мкм')
    minor_axis_length_microns = models.FloatField(null=True, blank=True, verbose_name='Длина малой оси (мкм)', help_text='Длина малой оси в мкм')
    
    # Position
    centroid_x = models.FloatField(verbose_name='Центроид X', help_text='X-координата центроида')
    centroid_y = models.FloatField(verbose_name='Центроид Y', help_text='Y-координата центроида')
    
    # Bounding box
    bounding_box_x = models.PositiveIntegerField(verbose_name='Охватывающий прямоугольник X', help_text='X-координата охватывающего прямоугольника')
    bounding_box_y = models.PositiveIntegerField(verbose_name='Охватывающий прямоугольник Y', help_text='Y-координата охватывающего прямоугольника')
    bounding_box_width = models.PositiveIntegerField(verbose_name='Ширина охватывающего прямоугольника', help_text='Ширина охватывающего прямоугольника')
    bounding_box_height = models.PositiveIntegerField(verbose_name='Высота охватывающего прямоугольника', help_text='Высота охватывающего прямоугольника')
    
    # GLCM Texture Features
    glcm_contrast = models.FloatField(null=True, blank=True, verbose_name='GLCM Контраст', help_text='Мера локальной вариации интенсивности')
    glcm_correlation = models.FloatField(null=True, blank=True, verbose_name='GLCM Корреляция', help_text='Мера линейной зависимости уровней серого')
    glcm_energy = models.FloatField(null=True, blank=True, verbose_name='GLCM Энергия', help_text='Мера текстурной однородности')
    glcm_homogeneity = models.FloatField(null=True, blank=True, verbose_name='GLCM Однородность', help_text='Мера близости распределения')
    glcm_entropy = models.FloatField(null=True, blank=True, verbose_name='GLCM Энтропия', help_text='Мера случайности')
    glcm_variance = models.FloatField(null=True, blank=True, verbose_name='GLCM Дисперсия', help_text='Мера неоднородности')
    glcm_sum_average = models.FloatField(null=True, blank=True, verbose_name='GLCM Сумма среднего', help_text='Сумма среднего GLCM')
    glcm_sum_variance = models.FloatField(null=True, blank=True, verbose_name='GLCM Сумма дисперсии', help_text='Сумма дисперсии GLCM')
    glcm_sum_entropy = models.FloatField(null=True, blank=True, verbose_name='GLCM Сумма энтропии', help_text='Сумма энтропии GLCM')
    glcm_difference_average = models.FloatField(null=True, blank=True, verbose_name='GLCM Разность среднего', help_text='Разность среднего GLCM')
    glcm_difference_variance = models.FloatField(null=True, blank=True, verbose_name='GLCM Разность дисперсии', help_text='Разность дисперсии GLCM')
    glcm_difference_entropy = models.FloatField(null=True, blank=True, verbose_name='GLCM Разность энтропии', help_text='Разность энтропии GLCM')
    
    # First-Order Statistical Features
    intensity_mean = models.FloatField(null=True, blank=True, verbose_name='Средняя интенсивность', help_text='Среднее значение интенсивности')
    intensity_std = models.FloatField(null=True, blank=True, verbose_name='Стд. откл. интенсивности', help_text='Стандартное отклонение интенсивности')
    intensity_variance = models.FloatField(null=True, blank=True, verbose_name='Дисперсия интенсивности', help_text='Дисперсия интенсивности')
    intensity_skewness = models.FloatField(null=True, blank=True, verbose_name='Асимметрия интенсивности', help_text='Асимметрия распределения интенсивности')
    intensity_kurtosis = models.FloatField(null=True, blank=True, verbose_name='Эксцесс интенсивности', help_text='Эксцесс распределения интенсивности')
    intensity_min = models.FloatField(null=True, blank=True, verbose_name='Мин. интенсивность', help_text='Минимальное значение интенсивности')
    intensity_max = models.FloatField(null=True, blank=True, verbose_name='Макс. интенсивность', help_text='Максимальное значение интенсивности')
    intensity_range = models.FloatField(null=True, blank=True, verbose_name='Диапазон интенсивности', help_text='Диапазон значений интенсивности')
    intensity_p10 = models.FloatField(null=True, blank=True, verbose_name='Интенсивность 10% перцентиль', help_text='10-й перцентиль интенсивности')
    intensity_p25 = models.FloatField(null=True, blank=True, verbose_name='Интенсивность 25% перцентиль', help_text='25-й перцентиль интенсивности')
    intensity_p75 = models.FloatField(null=True, blank=True, verbose_name='Интенсивность 75% перцентиль', help_text='75-й перцентиль интенсивности')
    intensity_p90 = models.FloatField(null=True, blank=True, verbose_name='Интенсивность 90% перцентиль', help_text='90-й перцентиль интенсивности')
    intensity_iqr = models.FloatField(null=True, blank=True, verbose_name='Интенсивность IQR', help_text='Межквартильный размах интенсивности')
    intensity_entropy = models.FloatField(null=True, blank=True, verbose_name='Энтропия интенсивности', help_text='Энтропия гистограммы интенсивности')
    intensity_energy = models.FloatField(null=True, blank=True, verbose_name='Энергия интенсивности', help_text='Энергия гистограммы интенсивности')
    intensity_median = models.FloatField(null=True, blank=True, verbose_name='Медиана интенсивности', help_text='Медианное значение интенсивности')
    intensity_mad = models.FloatField(null=True, blank=True, verbose_name='Мад интенсивности', help_text='Медианное абсолютное отклонение')
    intensity_cv = models.FloatField(null=True, blank=True, verbose_name='Коэф. вариации интенсивности', help_text='Коэффициент вариации')
    
    class Meta:
        ordering = ['cell_id']
        unique_together = ['analysis', 'cell_id']
    
    def __str__(self):
        return f"Cell {self.cell_id} from {self.analysis}"
