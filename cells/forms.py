from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Field, Fieldset, HTML, Div, Button
from .models import Cell, CellAnalysis, AnalysisBatch


class MultipleFileInput(forms.ClearableFileInput):
    """Custom widget for multiple file uploads"""
    allow_multiple_selected = True
    
    def __init__(self, attrs=None):
        if attrs is None:
            attrs = {}
        attrs.update({'multiple': True})
        super().__init__(attrs)
    
    def value_from_datadict(self, data, files, name):
        """Handle multiple file uploads"""
        if hasattr(files, 'getlist'):
            return files.getlist(name)
        else:
            return files.get(name)


class MultipleFileField(forms.FileField):
    """Custom field for multiple file uploads"""
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        """Clean multiple uploaded files"""
        # SingleValueField expects a single value, but we have multiple files
        if isinstance(data, (list, tuple)):
            cleaned_files = []
            for file in data:
                if file:  # Skip empty files
                    cleaned_file = super(forms.FileField, self).clean(file)
                    if cleaned_file:
                        cleaned_files.append(cleaned_file)
            return cleaned_files
        else:
            # Single file or None
            return super().clean(data, initial)


class CellUploadForm(forms.ModelForm):
    class Meta:
        model = Cell
        fields = ['name', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Введите название изображения'}),
            'image': forms.FileInput(attrs={'accept': 'image/*'}),
        }
        labels = {
            'name': 'Название изображения',
            'image': 'Изображение клетки',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            Field('name', css_class='form-control'),
            Field('image', css_class='form-control'),
            Submit('submit', 'Загрузить изображение клетки', css_class='btn btn-primary mt-3')
        )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Размер файла слишком большой (макс. 10МБ)")
            
            # Check file type
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError("Пожалуйста, загрузите корректный файл изображения")
            
            # Additional image quality checks
            try:
                from PIL import Image as PILImage
                import io
                
                # Try to open and verify the image
                image_data = image.read()
                image.seek(0)  # Reset file pointer
                
                pil_image = PILImage.open(io.BytesIO(image_data))
                
                # Check minimum dimensions
                if pil_image.width < 64 or pil_image.height < 64:
                    raise forms.ValidationError("Изображение слишком маленькое (минимум 64x64 пикселя)")
                
                # Check maximum dimensions (to prevent memory issues)
                if pil_image.width > 8192 or pil_image.height > 8192:
                    raise forms.ValidationError("Изображение слишком большое (максимум 8192x8192 пикселя)")
                
                # Check supported formats (including BMP)
                supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP']
                if pil_image.format not in supported_formats:
                    raise forms.ValidationError(f"Неподдерживаемый формат изображения '{pil_image.format}'. Поддерживаются: JPEG, PNG, TIFF, BMP")
                
                # Check if image is corrupted by trying to load pixel data
                pil_image.load()
                
                # Convert to RGB if needed for compatibility check
                if pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    raise forms.ValidationError("Неподдерживаемый режим изображения. Пожалуйста, используйте RGB, оттенки серого или палитровые изображения.")
                
            except Exception as e:
                if 'Image too small' in str(e) or 'Image too large' in str(e) or 'Unsupported image mode' in str(e):
                    raise  # Re-raise our custom validation errors
                else:
                    raise forms.ValidationError("Неверный или поврежденный файл изображения")
        
        return image


class CellAnalysisForm(forms.ModelForm):
    # Additional preprocessing fields (not in model, handled separately)
    apply_preprocessing = forms.BooleanField(
        required=False,
        label='Применить предобработку изображения',
        help_text='Включить продвинутую предобработку изображения для улучшения качества сегментации',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    apply_noise_reduction = forms.BooleanField(
        required=False,
        label='Подавление шума',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    noise_reduction_method = forms.ChoiceField(
        required=False,
        choices=[
            ('gaussian', 'Гауссов фильтр'),
            ('median', 'Медианный фильтр'),
            ('bilateral', 'Двусторонний фильтр (рекомендуется)'),
        ],
        initial='bilateral',
        label='Метод подавления шума',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    apply_contrast_enhancement = forms.BooleanField(
        required=False,
        label='Улучшение контраста',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    contrast_method = forms.ChoiceField(
        required=False,
        choices=[
            ('clahe', 'CLAHE (рекомендуется)'),
            ('histogram_eq', 'Гистограмная эквализация'),
            ('rescale', 'Масштабирование интенсивности'),
        ],
        initial='clahe',
        label='Метод контраста',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    apply_sharpening = forms.BooleanField(
        required=False,
        label='Повышение резкости изображения',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    apply_normalization = forms.BooleanField(
        required=False,
        label='Нормализация интенсивности',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    # Statistical Analysis Fields (TIER 1 CRITICAL ENHANCEMENT)
    enable_statistical_analysis = forms.BooleanField(
        required=False,
        initial=True,  # Включить по умолчанию для точных расчетов
        label='Включить статистический анализ',
        help_text='Добавить научный статистический анализ с доверительными интервалами и анализом неопределенности',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    confidence_level = forms.ChoiceField(
        required=False,
        choices=[
            (0.90, '90% доверительный интервал'),
            (0.95, '95% доверительный интервал (рекомендуется)'),
            (0.99, '99% доверительный интервал'),
        ],
        initial=0.95,
        label='Уровень доверия',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    bootstrap_samples = forms.ChoiceField(
        required=False,
        choices=[
            (1000, '1,000 итераций (быстро)'),
            (2000, '2,000 итераций (рекомендуется)'),
            (5000, '5,000 итераций (высокая точность)'),
        ],
        initial=2000,
        label='Bootstrap итерации',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = CellAnalysis
        fields = ['cellpose_model', 'cellpose_diameter', 'flow_threshold', 'cellprob_threshold', 'use_roi', 'filtering_mode']
        labels = {
            'cellpose_model': 'Модель Cellpose',
            'cellpose_diameter': 'Диаметр клетки (пиксели)',
            'flow_threshold': 'Порог потока',
            'cellprob_threshold': 'Порог вероятности клетки',
            'use_roi': 'Использовать выбор области интереса (ROI)',
            'filtering_mode': 'Режим фильтрации клеток',
        }
        widgets = {
            'cellpose_model': forms.Select(attrs={'class': 'form-select'}),
            'cellpose_diameter': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0',
                'placeholder': '30.0'
            }),
            'flow_threshold': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0',
                'max': '3',
                'placeholder': '0.4'
            }),
            'cellprob_threshold': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '-6',
                'max': '6',
                'placeholder': '0.0'
            }),
            'use_roi': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'filtering_mode': forms.Select(attrs={'class': 'form-select'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        # Simple layout without tab structure - template handles tabs
        self.helper.layout = Layout(
            # Basic fields - rendered in template
            Field('cellpose_model', css_class='form-select'),
            Field('cellpose_diameter', css_class='form-control'),
            Field('flow_threshold', css_class='form-control'),
            Field('cellprob_threshold', css_class='form-control'),
            Field('use_roi', css_class='form-check-input'),
            Field('filtering_mode', css_class='form-select'),
            
            # Preprocessing fields - rendered in template
            Field('apply_preprocessing', css_class='form-check-input'),
            Field('apply_noise_reduction', css_class='form-check-input'),
            Field('noise_reduction_method', css_class='form-select'),
            Field('apply_contrast_enhancement', css_class='form-check-input'),
            Field('contrast_method', css_class='form-select'),
            Field('apply_sharpening', css_class='form-check-input'),
            Field('apply_normalization', css_class='form-check-input'),
            
            # Statistical fields - rendered in template
            Field('enable_statistical_analysis', css_class='form-check-input'),
            Field('confidence_level', css_class='form-select'),
            Field('bootstrap_samples', css_class='form-select'),
            
            # Submit button
            HTML('<div class="form-actions mt-4 text-center">'),
                Submit('submit', 'Начать анализ', css_class='btn btn-success btn-lg px-5'),
                HTML('<a href="#" onclick="history.back()" class="btn btn-outline-secondary ms-3">Отмена</a>'),
            HTML('</div>')
        )
        
        # Add help text
        self.fields['cellpose_model'].help_text = "Выберите подходящую модель cellpose для ваших изображений"
        self.fields['cellpose_diameter'].help_text = "Ожидаемый диаметр клетки в пикселях (оставьте 0 для автоопределения)"
        self.fields['flow_threshold'].help_text = "Порог ошибки потока (0.4 по умолчанию, меньше = больше клеток)"
        self.fields['cellprob_threshold'].help_text = "Порог вероятности клетки (0.0 по умолчанию)"
        self.fields['use_roi'].help_text = "Отметьте, чтобы включить выбор области - вы можете нарисовать прямоугольники на изображении для фокусировки анализа"
        self.fields['apply_preprocessing'].help_text = "Включить продвинутую предобработку изображения для улучшения качества сегментации"
        self.fields['apply_noise_reduction'].help_text = "Уменьшить шум изображения для более четких границ клеток"
        self.fields['noise_reduction_method'].help_text = "Двусторонний фильтр лучше всего сохраняет края"
        self.fields['apply_contrast_enhancement'].help_text = "Улучшить контраст изображения для лучшей видимости клеток"
        self.fields['contrast_method'].help_text = "CLAHE лучше всего работает для микроскопических изображений"
        self.fields['apply_sharpening'].help_text = "Улучшить определение краев для лучшей сегментации"
        self.fields['apply_normalization'].help_text = "Стандартизировать значения интенсивности по всему изображению"
        self.fields['filtering_mode'].help_text = "Выберите строгость фильтрации: Никакой=сохранить все, Клинический=медицинский стандарт, Исследовательский=консервативный, Настраиваемый=продвинутые настройки"
        
        # Statistical analysis help text
        self.fields['enable_statistical_analysis'].help_text = "Включить научный статистический анализ с доверительными интервалами и анализом неопределенности для исследовательских целей"
        self.fields['confidence_level'].help_text = "Уровень доверия для доверительных интервалов (95% стандарт для научных публикаций)"
        self.fields['bootstrap_samples'].help_text = "Количество bootstrap итераций для статистического анализа (больше = точнее, но дольше)"
    
    def clean_cellpose_diameter(self):
        diameter = self.cleaned_data.get('cellpose_diameter')
        if diameter is not None and diameter < 0:
            raise forms.ValidationError("Диаметр должен быть положительным или 0 для автоопределения")
        return diameter
    
    def clean_flow_threshold(self):
        threshold = self.cleaned_data.get('flow_threshold')
        if threshold is not None and (threshold < 0 or threshold > 3):
            raise forms.ValidationError("Порог потока должен быть между 0 и 3")
        return threshold
    
    def clean_cellprob_threshold(self):
        threshold = self.cleaned_data.get('cellprob_threshold')
        if threshold is not None and (threshold < -6 or threshold > 6):
            raise forms.ValidationError("Порог вероятности клетки должен быть между -6 и 6")
        return threshold
    
    def clean(self):
        """
        Enhanced validation with parameter optimization recommendations.
        Validates parameter combinations and provides optimization hints.
        """
        cleaned_data = super().clean()
        
        # Import optimization utilities
        try:
            from .parameter_optimization import ParameterOptimizer
            
            # Validate parameter combinations
            params = {
                'cellpose_diameter': cleaned_data.get('cellpose_diameter'),
                'flow_threshold': cleaned_data.get('flow_threshold'),
                'cellprob_threshold': cleaned_data.get('cellprob_threshold'),
                'cellpose_model': cleaned_data.get('cellpose_model')
            }
            
            # Run parameter validation
            validation_errors = ParameterOptimizer.validate_parameters(params)
            
            # Add any validation errors to form
            for error in validation_errors:
                self.add_error(None, error)
            
            # Check for potentially problematic parameter combinations
            flow_threshold = cleaned_data.get('flow_threshold')
            cellprob_threshold = cleaned_data.get('cellprob_threshold')
            
            if flow_threshold is not None and cellprob_threshold is not None:
                if flow_threshold > 2.0 and cellprob_threshold > 3.0:
                    self.add_error(None, 
                        "Warning: High flow and probability thresholds may result in very few detected cells. "
                        "Consider using the Auto-Optimize feature for better parameter selection.")
                
                if flow_threshold < 0.2 and cellprob_threshold < -3.0:
                    self.add_error(None,
                        "Warning: Very low thresholds may result in over-segmentation and false positives. "
                        "Consider using the Auto-Optimize feature for more reliable results.")
        
        except ImportError:
            # Parameter optimization not available, skip enhanced validation
            pass
        except Exception as e:
            # Log error but don't fail form validation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Parameter validation failed: {str(e)}")
        
        return cleaned_data
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        # Build preprocessing options from form fields
        if self.cleaned_data.get('apply_preprocessing', False):
            preprocessing_options = {
                'apply_noise_reduction': self.cleaned_data.get('apply_noise_reduction', False),
                'noise_reduction_method': self.cleaned_data.get('noise_reduction_method', 'bilateral'),
                'apply_contrast_enhancement': self.cleaned_data.get('apply_contrast_enhancement', False),
                'contrast_method': self.cleaned_data.get('contrast_method', 'clahe'),
                'apply_sharpening': self.cleaned_data.get('apply_sharpening', False),
                'apply_normalization': self.cleaned_data.get('apply_normalization', False),
                'gaussian_sigma': 0.5,
                'median_disk_size': 2,
                'unsharp_radius': 1.0,
                'unsharp_amount': 1.0,
                'morphological_disk_size': 1,
            }
            instance.preprocessing_options = preprocessing_options
        else:
            instance.preprocessing_options = {}
        
        # Apply filtering preset based on selected mode
        filtering_mode = self.cleaned_data.get('filtering_mode', 'clinical')
        if filtering_mode != 'custom':  # Don't override custom settings
            instance.apply_filtering_preset(filtering_mode)
        
        # Build statistical analysis configuration
        if self.cleaned_data.get('enable_statistical_analysis', False):
            statistical_config = {
                'enable_statistical_analysis': True,
                'confidence_level': float(self.cleaned_data.get('confidence_level', 0.95)),
                'bootstrap_samples': int(self.cleaned_data.get('bootstrap_samples', 2000)),
                'include_confidence_intervals': True,
                'include_uncertainty_propagation': True,
                'include_bootstrap_analysis': True,
            }
            # Store in preprocessing_options for now (we'll add a dedicated field later if needed)
            if not instance.preprocessing_options:
                instance.preprocessing_options = {}
            instance.preprocessing_options['statistical_config'] = statistical_config
        else:
            # Ensure statistical analysis is disabled
            if instance.preprocessing_options and 'statistical_config' in instance.preprocessing_options:
                instance.preprocessing_options['statistical_config'] = {'enable_statistical_analysis': False}
        
        if commit:
            instance.save()
        return instance


class ScaleCalibrationForm(forms.Form):
    """Form for setting scale calibration"""
    reference_length_pixels = forms.FloatField(
        label='Референсная длина (пиксели)',
        help_text='Измерьте известный объект на изображении и введите длину в пикселях',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'placeholder': 'e.g., 150.5'
        })
    )
    
    reference_length_microns = forms.FloatField(
        label='Известная реальная длина (мкм)',
        help_text='Введите фактическую длину измеренного объекта в микронах',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'min': '0.01',
            'placeholder': 'e.g., 10.0'
        })
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_id = 'scale-calibration-form'
        self.helper.layout = Layout(
            Fieldset(
                'Калибровка масштаба',
                HTML('''
                    <div class="alert alert-info">
                        <h6><i class="fas fa-ruler"></i> How to calibrate scale:</h6>
                        <ol class="mb-0">
                            <li>Find a known object in your image (scale bar, cell culture dish, etc.)</li>
                            <li>Measure its length in pixels using the ruler tool above</li>
                            <li>Enter the pixel measurement and known real-world size</li>
                            <li>Click "Set Scale" to calibrate</li>
                        </ol>
                    </div>
                '''),
                Field('reference_length_pixels', css_class='form-control'),
                Field('reference_length_microns', css_class='form-control'),
            ),
            Submit('submit', 'Установить масштаб', css_class='btn btn-primary'),
            Button('cancel', 'Отмена', css_class='btn btn-secondary ms-2', onclick='hideScaleCalibration()')
        )
    
    def clean_reference_length_pixels(self):
        pixels = self.cleaned_data.get('reference_length_pixels')
        if pixels is not None and pixels <= 0:
            raise forms.ValidationError("Референсная длина в пикселях должна быть положительной")
        return pixels
    
    def clean_reference_length_microns(self):
        microns = self.cleaned_data.get('reference_length_microns')
        if microns is not None and microns <= 0:
            raise forms.ValidationError("Референсная длина в микронах должна быть положительной")
        return microns


class BatchCreateForm(forms.ModelForm):
    """Conservative form for creating analysis batches"""
    
    # Multiple file upload field
    images = MultipleFileField(
        widget=MultipleFileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        }),
        label='Изображения для пакетного анализа',
        help_text='Выберите до 10 изображений для анализа (макс. 10МБ каждое). Поддерживаются форматы: JPEG, PNG, TIFF, BMP.'
    )
    
    class Meta:
        model = AnalysisBatch
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Введите название пакета анализа'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Опциональное описание пакета анализа'
            }),
        }
        labels = {
            'name': 'Название пакета',
            'description': 'Описание',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            Fieldset(
                'Создание пакета анализа',
                HTML('''
                    <div class="alert alert-info">
                        <h6><i class="fas fa-layer-group"></i> Пакетный анализ</h6>
                        <p class="mb-0">Создайте пакет для анализа нескольких связанных изображений с одинаковыми параметрами.</p>
                        <small><strong>Ограничение RTX 2070 Max-Q:</strong> Максимум 10 изображений на пакет для оптимального использования памяти.</small>
                    </div>
                '''),
                Field('name', css_class='form-control'),
                Field('description', css_class='form-control'),
                Field('images', css_class='form-control'),
            ),
            Submit('submit', 'Создать пакет анализа', css_class='btn btn-primary')
        )
    
    def clean_images(self):
        """Validate uploaded images for batch processing"""
        # Get the cleaned data from the MultipleFileField
        images = self.cleaned_data.get('images')
        
        if not images:
            raise forms.ValidationError("Пожалуйста, выберите хотя бы одно изображение")
        
        # Handle both single file and list of files
        if not isinstance(images, list):
            images = [images]
        
        # Conservative limit for RTX 2070 Max-Q
        if len(images) > 10:
            raise forms.ValidationError("Максимум 10 изображений на пакет (ограничение GPU памяти)")
        
        # Validate each image
        for i, image in enumerate(images):
            # Check file size
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise forms.ValidationError(f"Изображение {i+1} слишком большое (макс. 10МБ)")
            
            # Check file type
            if hasattr(image, 'content_type') and not image.content_type.startswith('image/'):
                raise forms.ValidationError(f"Файл {i+1} не является изображением")
            
            # Basic PIL validation
            try:
                from PIL import Image as PILImage
                import io
                
                image_data = image.read()
                image.seek(0)  # Reset file pointer
                
                pil_image = PILImage.open(io.BytesIO(image_data))
                
                # Check dimensions (conservative for memory)
                if pil_image.width < 64 or pil_image.height < 64:
                    raise forms.ValidationError(f"Изображение {i+1} слишком маленькое (минимум 64x64)")
                
                if pil_image.width > 4096 or pil_image.height > 4096:
                    raise forms.ValidationError(f"Изображение {i+1} слишком большое (максимум 4096x4096 для пакетной обработки)")
                
                # Check supported formats (including BMP)
                supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP']
                if pil_image.format not in supported_formats:
                    raise forms.ValidationError(f"Изображение {i+1} имеет неподдерживаемый формат '{pil_image.format}'. Поддерживаются: JPEG, PNG, TIFF, BMP")
                
                pil_image.load()  # Verify image integrity
                
            except Exception as e:
                raise forms.ValidationError(f"Изображение {i+1} повреждено или неподдерживается")
        
        return images


class BatchAnalysisForm(forms.ModelForm):
    """Form for configuring batch analysis parameters"""
    
    # Shared analysis parameters (simplified version of CellAnalysisForm)
    cellpose_model = forms.ChoiceField(
        choices=CellAnalysis.CELLPOSE_MODEL_CHOICES,
        initial='cpsam',
        label='Модель Cellpose',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    cellpose_diameter = forms.FloatField(
        initial=0.0,  # Auto-detection
        label='Диаметр клетки (пиксели)',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0',
            'placeholder': '0 (автоопределение)'
        })
    )
    
    flow_threshold = forms.FloatField(
        initial=0.4,
        label='Порог потока',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0',
            'max': '3',
            'placeholder': '0.4'
        })
    )
    
    cellprob_threshold = forms.FloatField(
        initial=0.0,
        label='Порог вероятности клетки',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '-6',
            'max': '6',
            'placeholder': '0.0'
        })
    )
    
    filtering_mode = forms.ChoiceField(
        choices=CellAnalysis.FILTERING_MODE_CHOICES,
        initial='clinical',
        label='Режим фильтрации',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Conservative preprocessing options
    apply_preprocessing = forms.BooleanField(
        required=False,
        label='Применить базовую предобработку',
        help_text='Включить базовую предобработку для всех изображений в пакете',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    class Meta:
        model = AnalysisBatch
        fields = []  # We're not saving to the model directly
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.layout = Layout(
            Fieldset(
                'Параметры пакетного анализа',
                HTML('''
                    <div class="alert alert-warning">
                        <h6><i class="fas fa-memory"></i> RTX 2070 Max-Q оптимизация</h6>
                        <p class="mb-0">Параметры оптимизированы для вашей GPU. Изображения будут обрабатываться последовательно для предотвращения переполнения памяти.</p>
                    </div>
                '''),
                Field('cellpose_model', css_class='form-select'),
                Field('cellpose_diameter', css_class='form-control'),
                Field('flow_threshold', css_class='form-control'),
                Field('cellprob_threshold', css_class='form-control'),
                Field('filtering_mode', css_class='form-select'),
                Field('apply_preprocessing', css_class='form-check-input'),
            ),
            Submit('submit', 'Начать пакетный анализ', css_class='btn btn-success btn-lg')
        )
        
        # Add help text
        self.fields['cellpose_model'].help_text = "Модель для всех изображений в пакете"
        self.fields['cellpose_diameter'].help_text = "0 = автоопределение для каждого изображения"
        self.fields['flow_threshold'].help_text = "Консервативное значение для стабильных результатов"
        self.fields['cellprob_threshold'].help_text = "Стандартное значение для большинства случаев"
        self.fields['filtering_mode'].help_text = "Режим фильтрации для всех изображений"
        self.fields['apply_preprocessing'].help_text = "Базовая предобработка для улучшения качества"
    
    def get_analysis_parameters(self):
        """Return analysis parameters as dict for batch processing"""
        return {
            'cellpose_model': self.cleaned_data['cellpose_model'],
            'cellpose_diameter': self.cleaned_data['cellpose_diameter'],
            'flow_threshold': self.cleaned_data['flow_threshold'],
            'cellprob_threshold': self.cleaned_data['cellprob_threshold'],
            'filtering_mode': self.cleaned_data['filtering_mode'],
            'apply_preprocessing': self.cleaned_data['apply_preprocessing'],
            'use_roi': False,  # Disabled for batch processing simplicity
            # Conservative preprocessing options
            'preprocessing_options': {
                'apply_noise_reduction': True if self.cleaned_data['apply_preprocessing'] else False,
                'noise_reduction_method': 'bilateral',
                'apply_contrast_enhancement': True if self.cleaned_data['apply_preprocessing'] else False,
                'contrast_method': 'clahe',
                'apply_sharpening': False,  # Disabled for batch consistency
                'apply_normalization': True if self.cleaned_data['apply_preprocessing'] else False,
            }
        }