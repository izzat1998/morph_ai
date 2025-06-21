from django import forms
from django.utils.translation import gettext_lazy as _
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Field, Fieldset, HTML, Div, Button
from .models import Cell, CellAnalysis


class CellUploadForm(forms.ModelForm):
    class Meta:
        model = Cell
        fields = ['name', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': _('Enter image name')}),
            'image': forms.FileInput(attrs={'accept': 'image/*'}),
        }
        labels = {
            'name': _('Image Name'),
            'image': _('Cell Image'),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            Field('name', css_class='form-control'),
            Field('image', css_class='form-control'),
            Submit('submit', _('Upload Cell Image'), css_class='btn btn-primary mt-3')
        )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError(_("Image file too large (max 10MB)"))
            
            # Check file type
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError(_("Please upload a valid image file"))
            
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
                    raise forms.ValidationError(_("Image too small (minimum 64x64 pixels)"))
                
                # Check maximum dimensions (to prevent memory issues)
                if pil_image.width > 8192 or pil_image.height > 8192:
                    raise forms.ValidationError(_("Image too large (maximum 8192x8192 pixels)"))
                
                # Check if image is corrupted by trying to load pixel data
                pil_image.load()
                
                # Convert to RGB if needed for compatibility check
                if pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    raise forms.ValidationError(_("Unsupported image mode. Please use RGB, grayscale, or palette images."))
                
            except Exception as e:
                if 'Image too small' in str(e) or 'Image too large' in str(e) or 'Unsupported image mode' in str(e):
                    raise  # Re-raise our custom validation errors
                else:
                    raise forms.ValidationError(_("Invalid or corrupted image file"))
        
        return image


class CellAnalysisForm(forms.ModelForm):
    # Additional preprocessing fields (not in model, handled separately)
    apply_preprocessing = forms.BooleanField(
        required=False,
        label=_('Apply Image Preprocessing'),
        help_text=_('Enable advanced image preprocessing to improve segmentation quality'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    apply_noise_reduction = forms.BooleanField(
        required=False,
        label=_('Noise Reduction'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    noise_reduction_method = forms.ChoiceField(
        required=False,
        choices=[
            ('gaussian', _('Gaussian Filter')),
            ('median', _('Median Filter')),
            ('bilateral', _('Bilateral Filter (Recommended)')),
        ],
        initial='bilateral',
        label=_('Noise Reduction Method'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    apply_contrast_enhancement = forms.BooleanField(
        required=False,
        label=_('Contrast Enhancement'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    contrast_method = forms.ChoiceField(
        required=False,
        choices=[
            ('clahe', _('CLAHE (Recommended)')),
            ('histogram_eq', _('Histogram Equalization')),
            ('rescale', _('Intensity Rescaling')),
        ],
        initial='clahe',
        label=_('Contrast Method'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    apply_sharpening = forms.BooleanField(
        required=False,
        label=_('Image Sharpening'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    apply_normalization = forms.BooleanField(
        required=False,
        label=_('Intensity Normalization'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    class Meta:
        model = CellAnalysis
        fields = ['cellpose_model', 'cellpose_diameter', 'flow_threshold', 'cellprob_threshold', 'use_roi', 'filtering_mode']
        labels = {
            'cellpose_model': _('Cellpose Model'),
            'cellpose_diameter': _('Cell Diameter (pixels)'),
            'flow_threshold': _('Flow Threshold'),
            'cellprob_threshold': _('Cell Probability Threshold'),
            'use_roi': _('Use Region of Interest (ROI) Selection'),
            'filtering_mode': _('Cell Filtering Mode'),
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
        self.helper.layout = Layout(
            # Tab structure with organized sections
            HTML('''
                <div class="tab-content" id="configTabContent">
                    <div class="tab-pane fade show active" id="basic-config" role="tabpanel">
                        <div class="basic-config-section">
            '''),
            Fieldset(
                _('Core Parameters'),
                HTML('<div class="row g-3">'),
                    HTML('<div class="col-md-6">'),
                        Field('cellpose_model', css_class='form-select'),
                    HTML('</div>'),
                    HTML('<div class="col-md-6">'),
                        Field('cellpose_diameter', css_class='form-control'),
                    HTML('</div>'),
                HTML('</div>'),
                HTML('<div class="row g-3 mt-3">'),
                    HTML('<div class="col-md-6">'),
                        Field('flow_threshold', css_class='form-control'),
                    HTML('</div>'),
                    HTML('<div class="col-md-6">'),
                        Field('cellprob_threshold', css_class='form-control'),
                    HTML('</div>'),
                HTML('</div>'),
                HTML('<div class="mt-3">'),
                    Div(
                        Field('use_roi', css_class='form-check-input'),
                        HTML('<small class="text-muted d-block mt-1">' + str(_('Enable to select specific regions for analysis')) + '</small>'),
                        css_class='form-check'
                    ),
                HTML('</div>'),
            ),
            HTML('''
                        </div>
                    </div>
                    <div class="tab-pane fade" id="advanced-config" role="tabpanel">
            '''),
            Fieldset(
                _('Image Preprocessing (Optional)'),
                HTML('<p class="text-muted small">' + str(_('Advanced preprocessing options to improve image quality before segmentation.')) + '</p>'),
                Div(
                    Field('apply_preprocessing', css_class='form-check-input', id='id_apply_preprocessing'),
                    HTML('<small class="text-muted">' + str(_('Enable preprocessing options below')) + '</small>'),
                    css_class='form-check mb-3'
                ),
                HTML('<div id="preprocessing-options" style="display: none;" class="preprocessing-grid">'),
                    HTML('<div class="row g-3">'),
                        HTML('<div class="col-md-6">'),
                            Div(
                                Field('apply_noise_reduction', css_class='form-check-input'),
                                HTML('<label class="form-check-label fw-bold">' + str(_('Noise Reduction')) + '</label>'),
                                css_class='form-check mb-2'
                            ),
                            Field('noise_reduction_method', css_class='form-select form-select-sm'),
                        HTML('</div>'),
                        HTML('<div class="col-md-6">'),
                            Div(
                                Field('apply_contrast_enhancement', css_class='form-check-input'),
                                HTML('<label class="form-check-label fw-bold">' + str(_('Contrast Enhancement')) + '</label>'),
                                css_class='form-check mb-2'
                            ),
                            Field('contrast_method', css_class='form-select form-select-sm'),
                        HTML('</div>'),
                    HTML('</div>'),
                    HTML('<div class="row g-3 mt-2">'),
                        HTML('<div class="col-md-6">'),
                            Div(
                                Field('apply_sharpening', css_class='form-check-input'),
                                HTML('<label class="form-check-label fw-bold">' + str(_('Image Sharpening')) + '</label>'),
                                css_class='form-check'
                            ),
                        HTML('</div>'),
                        HTML('<div class="col-md-6">'),
                            Div(
                                Field('apply_normalization', css_class='form-check-input'),
                                HTML('<label class="form-check-label fw-bold">' + str(_('Intensity Normalization')) + '</label>'),
                                css_class='form-check'
                            ),
                        HTML('</div>'),
                    HTML('</div>'),
                HTML('</div>'),
                HTML('''
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const preprocessingCheckbox = document.getElementById('id_apply_preprocessing');
                    const preprocessingOptions = document.getElementById('preprocessing-options');
                    
                    function togglePreprocessingOptions() {
                        if (preprocessingCheckbox.checked) {
                            preprocessingOptions.style.display = 'block';
                        } else {
                            preprocessingOptions.style.display = 'none';
                        }
                    }
                    
                    preprocessingCheckbox.addEventListener('change', togglePreprocessingOptions);
                    togglePreprocessingOptions(); // Set initial state
                });
                </script>
                '''),
            ),
            HTML('''
                    </div>
                    <div class="tab-pane fade" id="filtering-config" role="tabpanel">
            '''),
            Fieldset(
                _('Cell Filtering Options'),
                HTML('<p class="text-muted small">' + str(_('Control how strictly cells are filtered during analysis. Choose based on your use case.')) + '</p>'),
                Field('filtering_mode', css_class='form-select'),
                HTML('''
                <div class="filtering-guide mt-3 p-3 bg-light rounded">
                    <h6 class="text-primary mb-2">Filtering Mode Guide</h6>
                    <div class="row g-2">
                        <div class="col-md-6">
                            <div class="guide-item p-2 border rounded bg-white">
                                <strong class="text-success">No Filtering</strong><br>
                                <small class="text-muted">Keep all detected cells</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="guide-item p-2 border rounded bg-white">
                                <strong class="text-info">Basic</strong><br>
                                <small class="text-muted">Remove only obvious artifacts</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="guide-item p-2 border rounded bg-white">
                                <strong class="text-warning">Research</strong><br>
                                <small class="text-muted">Conservative filtering for research</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="guide-item p-2 border rounded bg-white">
                                <strong class="text-primary">Clinical</strong><br>
                                <small class="text-muted">Standard medical-grade filtering</small>
                            </div>
                        </div>
                    </div>
                </div>
                '''),
            ),
            HTML('''
                    </div>
                </div>
            '''),
            HTML('<div class="form-actions mt-4 text-center">'),
                Submit('submit', _('Start Analysis'), css_class='btn btn-success btn-lg px-5'),
                HTML('<a href="#" onclick="history.back()" class="btn btn-outline-secondary ms-3">' + str(_('Cancel')) + '</a>'),
            HTML('</div>')
        )
        
        # Add help text
        self.fields['cellpose_model'].help_text = _("Choose the appropriate cellpose model for your images")
        self.fields['cellpose_diameter'].help_text = _("Expected cell diameter in pixels (leave 0 for auto-detection)")
        self.fields['flow_threshold'].help_text = _("Flow error threshold (0.4 is default, lower = more cells)")
        self.fields['cellprob_threshold'].help_text = _("Cell probability threshold (0.0 is default)")
        self.fields['use_roi'].help_text = _("Check to enable region selection - you can draw rectangles on the image to focus analysis")
        self.fields['apply_preprocessing'].help_text = _("Enable advanced image preprocessing to improve segmentation quality")
        self.fields['apply_noise_reduction'].help_text = _("Reduce image noise for clearer cell boundaries")
        self.fields['noise_reduction_method'].help_text = _("Bilateral filter preserves edges best")
        self.fields['apply_contrast_enhancement'].help_text = _("Improve image contrast for better cell visibility")
        self.fields['contrast_method'].help_text = _("CLAHE works best for microscopy images")
        self.fields['apply_sharpening'].help_text = _("Enhance edge definition for better segmentation")
        self.fields['apply_normalization'].help_text = _("Standardize intensity values across the image")
        self.fields['filtering_mode'].help_text = _("Choose filtering strictness: None=keep all, Clinical=medical standard, Research=conservative, Custom=advanced settings")
    
    def clean_cellpose_diameter(self):
        diameter = self.cleaned_data.get('cellpose_diameter')
        if diameter is not None and diameter < 0:
            raise forms.ValidationError(_("Diameter must be positive or 0 for auto-detection"))
        return diameter
    
    def clean_flow_threshold(self):
        threshold = self.cleaned_data.get('flow_threshold')
        if threshold is not None and (threshold < 0 or threshold > 3):
            raise forms.ValidationError(_("Flow threshold must be between 0 and 3"))
        return threshold
    
    def clean_cellprob_threshold(self):
        threshold = self.cleaned_data.get('cellprob_threshold')
        if threshold is not None and (threshold < -6 or threshold > 6):
            raise forms.ValidationError(_("Cell probability threshold must be between -6 and 6"))
        return threshold
    
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
        
        if commit:
            instance.save()
        return instance


class ScaleCalibrationForm(forms.Form):
    """Form for setting scale calibration"""
    reference_length_pixels = forms.FloatField(
        label=_('Reference Length (pixels)'),
        help_text=_('Measure a known object in the image and enter the length in pixels'),
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'placeholder': 'e.g., 150.5'
        })
    )
    
    reference_length_microns = forms.FloatField(
        label=_('Known Real Length (μm)'),
        help_text=_('Enter the actual length of the measured object in microns'),
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
                _('Scale Calibration'),
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
            Submit('submit', _('Set Scale'), css_class='btn btn-primary'),
            Button('cancel', _('Cancel'), css_class='btn btn-secondary ms-2', onclick='hideScaleCalibration()')
        )
    
    def clean_reference_length_pixels(self):
        pixels = self.cleaned_data.get('reference_length_pixels')
        if pixels is not None and pixels <= 0:
            raise forms.ValidationError(_("Reference length in pixels must be positive"))
        return pixels
    
    def clean_reference_length_microns(self):
        microns = self.cleaned_data.get('reference_length_microns')
        if microns is not None and microns <= 0:
            raise forms.ValidationError(_("Reference length in microns must be positive"))
        return microns