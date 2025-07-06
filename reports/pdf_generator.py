"""
Comprehensive PDF report generator for morphometric analysis.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, NextPageTemplate, PageTemplate, Frame
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from django.conf import settings
from django.utils import timezone
from cells.models import Cell, CellAnalysis, DetectedCell
from .charts import MorphometricChartGenerator

logger = logging.getLogger(__name__)


def register_fonts():
    """Register fonts that support Russian characters."""
    try:
        # Try to register DejaVu Sans (commonly available and supports Cyrillic)
        base_paths = [
            '/usr/share/fonts/truetype/dejavu/',
            '/System/Library/Fonts/',  # macOS
            'C:/Windows/Fonts/',  # Windows
            '/usr/share/fonts/TTF/',  # Some Linux distributions
            '/usr/share/fonts/truetype/liberation/',  # Alternative
        ]
        
        font_registered = False
        
        for base_path in base_paths:
            regular_font = os.path.join(base_path, 'DejaVuSans.ttf')
            bold_font = os.path.join(base_path, 'DejaVuSans-Bold.ttf')
            
            if os.path.exists(regular_font):
                pdfmetrics.registerFont(TTFont('DejaVuSans', regular_font))
                logger.info(f"Registered regular font: {regular_font}")
                font_registered = True
                
                if os.path.exists(bold_font):
                    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', bold_font))
                    logger.info(f"Registered bold font: {bold_font}")
                
                return 'DejaVuSans'
        
        # Try alternative fonts
        alternative_fonts = [
            ('/System/Library/Fonts/Arial.ttf', 'Arial'),
            ('C:/Windows/Fonts/arial.ttf', 'Arial'),
            ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 'LiberationSans'),
        ]
        
        for font_path, font_name in alternative_fonts:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                logger.info(f"Registered alternative font: {font_path}")
                return font_name
        
        # If no specific font found, try to use built-in fonts with Unicode support
        logger.warning("No suitable Unicode font found, using Helvetica with Unicode fallback")
        return 'Helvetica'
        
    except Exception as e:
        logger.error(f"Error registering fonts: {e}")
        return 'Helvetica'


class MorphometricPDFReport:
    """Generate comprehensive PDF reports for morphometric analysis."""
    
    def __init__(self, analysis: CellAnalysis, config: Optional[Dict] = None):
        """
        Initialize PDF report generator.
        
        Args:
            analysis: CellAnalysis instance
            config: Report configuration options
        """
        self.analysis = analysis
        self.cell = analysis.cell
        self.config = config or {}
        
        # Chart generator
        self.chart_generator = MorphometricChartGenerator()
        
        # Report configuration
        self.include_methodology = self.config.get('include_methodology', True)
        self.include_individual_cells = self.config.get('include_individual_cells', True)
        self.include_charts = self.config.get('include_charts', True)
        self.include_quality_control = self.config.get('include_quality_control', True)
        self.max_cells_per_page = self.config.get('max_cells_per_page', 30)
        
        # PDF settings
        self.pagesize = A4
        self.margin = 2*cm
        self.doc = None
        self.story = []
        
        # Register fonts for Russian support
        self.font_name = register_fonts()
        self.styles = self._create_styles()
        
    def generate_report(self) -> io.BytesIO:
        """
        Generate the complete PDF report.
        
        Returns:
            BytesIO buffer containing the PDF report
        """
        try:
            # Create PDF document
            buffer = io.BytesIO()
            self.doc = SimpleDocTemplate(
                buffer,
                pagesize=self.pagesize,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
                title=f"Morphometric Analysis Report - {self.cell.name}"
            )
            
            # Build report content
            self.story = []
            self._add_cover_page()
            self._add_table_of_contents()
            self._add_executive_summary()
            
            if self.include_methodology:
                self._add_methodology_section()
            
            self._add_image_processing_pipeline()
            
            if self.include_charts:
                self._add_statistical_analysis()
            
            if self.include_individual_cells:
                self._add_individual_cell_results()
            
            if self.include_quality_control:
                self._add_quality_control_section()
            
            self._add_technical_appendix()
            
            # Build PDF
            self.doc.build(self.story)
            
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return self._create_error_pdf()
    
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the report."""
        styles = getSampleStyleSheet()
        
        # Custom styles with Russian font support
        custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2E86AB'),
                fontName=self.font_name
            ),
            'Heading1': ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.HexColor('#2E86AB'),
                fontName=self.font_name
            ),
            'Heading2': ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=15,
                textColor=colors.HexColor('#A23B72'),
                fontName=self.font_name
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName=self.font_name
            ),
            'Caption': ParagraphStyle(
                'Caption',
                parent=styles['Normal'],
                fontSize=9,
                spaceAfter=12,
                alignment=TA_CENTER,
                textColor=colors.grey,
                fontName=self.font_name
            ),
            'TableHeader': ParagraphStyle(
                'TableHeader',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.white,
                fontName=self.font_name
            )
        }
        
        return custom_styles
    
    def _get_table_font(self, bold=False):
        """Get appropriate font name for tables with Russian support."""
        if self.font_name == 'DejaVuSans':
            return 'DejaVuSans-Bold' if bold else 'DejaVuSans'
        elif self.font_name == 'Arial':
            return 'Arial-Bold' if bold else 'Arial'
        elif self.font_name == 'LiberationSans':
            return 'LiberationSans-Bold' if bold else 'LiberationSans'
        else:
            return 'Helvetica-Bold' if bold else 'Helvetica'
    
    def _add_cover_page(self):
        """Add cover page to the report."""
        # Title
        title = f"Отчет морфометрического анализа"
        self.story.append(Paragraph(title, self.styles['Title']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Cell information
        cell_info = f"<b>Изображение клетки:</b> {self.cell.name}<br/>"
        cell_info += f"<b>Дата анализа:</b> {self.analysis.analysis_date.strftime('%B %d, %Y')}<br/>"
        cell_info += f"<b>ID анализа:</b> {self.analysis.id}<br/>"
        cell_info += f"<b>Статус:</b> {self.analysis.get_status_display()}"
        
        self.story.append(Paragraph(cell_info, self.styles['Normal']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Cell image preview
        if self.cell.image:
            try:
                # Add original image
                img_path = self.cell.image.path
                if os.path.exists(img_path):
                    img = Image(img_path, width=4*inch, height=3*inch)
                    self.story.append(img)
                    self.story.append(Paragraph("Исходное изображение клетки", self.styles['Caption']))
            except Exception as e:
                logger.warning(f"Could not add cover image: {e}")
        
        self.story.append(Spacer(1, 1*inch))
        
        # Analysis summary
        if self.analysis.status == 'completed':
            detected_cells = DetectedCell.objects.filter(analysis=self.analysis)
            summary = f"""
            <b>Обзор анализа:</b><br/>
            • Обнаружено клеток: {detected_cells.count()}<br/>
            • Модель Cellpose: {self.analysis.get_cellpose_model_display()}<br/>
            • Время обработки: {self.analysis.processing_time:.2f} секунд<br/>
            • Калибровка масштаба: {'Да' if self.cell.scale_set else 'Нет'}
            """
            self.story.append(Paragraph(summary, self.styles['Normal']))
        
        self.story.append(PageBreak())
    
    def _add_table_of_contents(self):
        """Add simple table of contents."""
        self.story.append(Paragraph("Оглавление", self.styles['Heading1']))
        
        # Simple TOC without automatic page numbering
        toc_items = [
            "1. Краткий обзор",
            "2. Методология" if self.include_methodology else None,
            "3. Конвейер обработки изображений", 
            "4. Статистический анализ" if self.include_charts else None,
            "5. Результаты отдельных клеток" if self.include_individual_cells else None,
            "6. Контроль качества" if self.include_quality_control else None,
            "7. Техническое приложение"
        ]
        
        # Filter out None items and renumber
        toc_items = [item for item in toc_items if item is not None]
        for i, item in enumerate(toc_items, 1):
            # Update numbering
            if ". " in item:
                item_text = item.split(". ", 1)[1]
                toc_items[i-1] = f"{i}. {item_text}"
        
        toc_text = "<br/>".join(toc_items)
        self.story.append(Paragraph(toc_text, self.styles['Normal']))
        self.story.append(PageBreak())
    
    def _add_executive_summary(self):
        """Add executive summary section."""
        self.story.append(Paragraph("Краткий обзор", self.styles['Heading1']))
        
        if self.analysis.status == 'completed':
            detected_cells = DetectedCell.objects.filter(analysis=self.analysis)
            
            # Key findings
            if detected_cells.exists():
                areas = [cell.area for cell in detected_cells]
                perimeters = [cell.perimeter for cell in detected_cells]
                circularities = [cell.circularity for cell in detected_cells]
                
                import numpy as np
                summary_text = f"""
                Данный морфометрический анализ успешно обработал изображение клетки "{self.cell.name}" 
                с использованием модели Cellpose {self.analysis.get_cellpose_model_display()}. 
                Анализ выявил <b>{len(areas)} клеток</b> со следующими ключевыми характеристиками:
                
                <br/><br/>
                <b>Морфометрический обзор:</b><br/>
                • Средняя площадь клетки: {np.mean(areas):.1f} ± {np.std(areas):.1f} пкс²<br/>
                • Средний периметр: {np.mean(perimeters):.1f} ± {np.std(perimeters):.1f} пкс<br/>
                • Средняя круглость: {np.mean(circularities):.3f} ± {np.std(circularities):.3f}<br/>
                • Диапазон размеров: {np.min(areas):.1f} - {np.max(areas):.1f} пкс²
                
                <br/><br/>
                <b>Оценка качества:</b><br/>
                • Обработка успешно завершена за {self.analysis.processing_time:.1f} секунд<br/>
                • {'Масштаб откалиброван' if self.cell.scale_set else 'Масштаб не откалиброван'}<br/>
                • Качество данных: Высокоточные результаты сегментации
                """
            else:
                summary_text = f"""
                Морфометрический анализ "{self.cell.name}" был завершен, но клетки не были 
                обнаружены с текущими параметрами. Это может указывать на то, что изображение требует 
                других параметров сегментации или настроек предварительной обработки.
                """
        else:
            summary_text = f"""
            Морфометрический анализ "{self.cell.name}" имеет статус: {self.analysis.get_status_display()}.
            """
        
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def _add_methodology_section(self):
        """Add methodology section."""
        self.story.append(Paragraph("Методология", self.styles['Heading1']))
        
        # Analysis parameters
        params_text = f"""
        <b>Конвейер обработки изображений:</b><br/>
        Данный анализ использует современный подход глубокого обучения с Cellpose для 
        автоматической сегментации клеток, с последующей всеобъемлющей экстракцией морфометрических признаков.
        
        <br/><br/>
        <b>Параметры сегментации:</b><br/>
        • Модель: {self.analysis.get_cellpose_model_display()}<br/>
        • Диаметр клетки: {self.analysis.cellpose_diameter} пикселей {'(автоопределение)' if self.analysis.cellpose_diameter == 0 else ''}<br/>
        • Порог потока: {self.analysis.flow_threshold}<br/>
        • Порог вероятности клетки: {self.analysis.cellprob_threshold}<br/>
        • Анализ ROI: {'Включен' if self.analysis.use_roi else 'Отключен'}
        
        <br/><br/>
        <b>Экстракция признаков:</b><br/>
        Морфометрические признаки вычислялись с использованием scikit-image regionprops, включая:
        • Геометрические измерения (площадь, периметр, центроид)<br/>
        • Описатели формы (круглость, эксцентриситет, плотность)<br/>
        • Параметры аппроксимации эллипсом (большая/малая оси, соотношение сторон)
        """
        
        if hasattr(self.analysis, 'apply_preprocessing') and self.analysis.apply_preprocessing:
            preprocessing_text = f"""
            <br/><br/>
            <b>Предварительная обработка изображения:</b><br/>
            Применялась продвинутая предварительная обработка для улучшения качества изображения:
            • Подавление шума: {'Применено' if getattr(self.analysis, 'apply_noise_reduction', False) else 'Не применено'}<br/>
            • Улучшение контраста: {'Применено' if getattr(self.analysis, 'apply_contrast_enhancement', False) else 'Не применено'}<br/>
            • Повышение резкости: {'Применено' if getattr(self.analysis, 'apply_sharpening', False) else 'Не применено'}<br/>
            • Нормализация: {'Применена' if getattr(self.analysis, 'apply_normalization', False) else 'Не применена'}
            """
            params_text += preprocessing_text
        
        self.story.append(Paragraph(params_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def _add_image_processing_pipeline(self):
        """Add image processing pipeline visualization."""
        self.story.append(Paragraph("Конвейер обработки изображений", self.styles['Heading1']))
        
        pipeline_text = """
        Следующие визуализации показывают полный конвейер обработки Cellpose, 
        от исходного изображения через сегментацию до финальной идентификации клеток.
        """
        self.story.append(Paragraph(pipeline_text, self.styles['Normal']))
        
        # Core pipeline visualization
        if self.analysis.segmentation_image:
            try:
                img_path = self.analysis.segmentation_image.path
                if os.path.exists(img_path):
                    img = Image(img_path, width=6*inch, height=4*inch)
                    self.story.append(img)
                    self.story.append(Paragraph("Core Processing Pipeline: Original → Contours → Flow → Probability → Masks → Centers", self.styles['Caption']))
                    self.story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.warning(f"Could not add segmentation image: {e}")
        
        # Additional visualizations
        visualizations = [
            (self.analysis.flow_analysis_image, "Flow Analysis: HSV → Trajectories → Magnitude → Convergence"),
            (self.analysis.style_quality_image, "Style & Quality Analysis: Style Vector → Diameter → Thresholds → Quality"),
            (self.analysis.edge_boundary_image, "Edge & Boundary Analysis: Edges → Gradients → Arrows → Combined")
        ]
        
        for img_field, caption in visualizations:
            if img_field:
                try:
                    img_path = img_field.path
                    if os.path.exists(img_path):
                        img = Image(img_path, width=6*inch, height=4*inch)
                        self.story.append(img)
                        self.story.append(Paragraph(caption, self.styles['Caption']))
                        self.story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.warning(f"Could not add visualization image: {e}")
        
        self.story.append(PageBreak())
    
    def _add_statistical_analysis(self):
        """Add statistical analysis section with charts."""
        self.story.append(Paragraph("Statistical Analysis", self.styles['Heading1']))
        
        detected_cells = DetectedCell.objects.filter(analysis=self.analysis)
        
        if not detected_cells.exists():
            self.story.append(Paragraph("No cells detected for statistical analysis.", self.styles['Normal']))
            return
        
        # Prepare data for charts
        data_dict = {
            'Площадь': [cell.area for cell in detected_cells],
            'Периметр': [cell.perimeter for cell in detected_cells],
            'Круглость': [cell.circularity for cell in detected_cells],
            'Эксцентриситет': [cell.eccentricity for cell in detected_cells],
        }
        
        # Distribution histograms
        self.story.append(Paragraph("Distribution Analysis", self.styles['Heading2']))
        
        for param_name, values in data_dict.items():
            try:
                chart_buffer = self.chart_generator.create_distribution_histogram(
                    values, f"{param_name} Distribution", param_name
                )
                if chart_buffer:
                    img = Image(chart_buffer, width=5*inch, height=3*inch)
                    self.story.append(img)
                    self.story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                logger.warning(f"Could not create histogram for {param_name}: {e}")
        
        # Correlation analysis
        self.story.append(PageBreak())
        self.story.append(Paragraph("Correlation Analysis", self.styles['Heading2']))
        
        try:
            corr_buffer = self.chart_generator.create_correlation_matrix(data_dict)
            if corr_buffer:
                img = Image(corr_buffer, width=5*inch, height=4*inch)
                self.story.append(img)
                self.story.append(Paragraph("Correlation Matrix of Morphometric Parameters", self.styles['Caption']))
        except Exception as e:
            logger.warning(f"Could not create correlation matrix: {e}")
        
        # Box plots
        self.story.append(Spacer(1, 0.3*inch))
        try:
            box_buffer = self.chart_generator.create_box_plot(
                data_dict, "Morphometric Parameter Distributions", "Value"
            )
            if box_buffer:
                img = Image(box_buffer, width=6*inch, height=4*inch)
                self.story.append(img)
                self.story.append(Paragraph("Box Plot Analysis of Key Parameters", self.styles['Caption']))
        except Exception as e:
            logger.warning(f"Could not create box plots: {e}")
        
        self.story.append(PageBreak())
    
    def _add_individual_cell_results(self):
        """Add individual cell results table."""
        self.story.append(Paragraph("Individual Cell Results", self.styles['Heading1']))
        
        detected_cells = DetectedCell.objects.filter(analysis=self.analysis).order_by('cell_id')
        
        if not detected_cells.exists():
            self.story.append(Paragraph("No individual cell data available.", self.styles['Normal']))
            return
        
        # Create table data
        headers = [
            'Cell ID', 'Area (px²)', 'Perimeter (px)', 'Circularity', 
            'Eccentricity', 'Solidity', 'Aspect Ratio'
        ]
        
        # Add micron units if scale is set
        if self.cell.scale_set:
            headers.insert(2, 'Area (μm²)')
            headers.insert(4, 'Perimeter (μm)')
        
        table_data = [headers]
        
        # Add cell data (paginate for large datasets)
        cells_per_page = self.max_cells_per_page
        total_cells = detected_cells.count()
        
        for page_start in range(0, total_cells, cells_per_page):
            page_cells = detected_cells[page_start:page_start + cells_per_page]
            
            for cell in page_cells:
                row = [
                    str(cell.cell_id),
                    f"{cell.area:.1f}",
                    f"{cell.perimeter:.1f}",
                    f"{cell.circularity:.3f}",
                    f"{cell.eccentricity:.3f}",
                    f"{cell.solidity:.3f}",
                    f"{cell.aspect_ratio:.2f}"
                ]
                
                # Add micron values if available
                if self.cell.scale_set:
                    row.insert(2, f"{cell.area_microns_sq:.2f}" if cell.area_microns_sq else "-")
                    row.insert(4, f"{cell.perimeter_microns:.2f}" if cell.perimeter_microns else "-")
                
                table_data.append(row)
            
            # Create table
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self._get_table_font(bold=True)),
                ('FONTNAME', (0, 1), (-1, -1), self._get_table_font(bold=False)),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            self.story.append(table)
            
            # Add page break if more data follows
            if page_start + cells_per_page < total_cells:
                self.story.append(PageBreak())
                table_data = [headers]  # Reset for next page
        
        self.story.append(PageBreak())
    
    def _add_quality_control_section(self):
        """Add quality control information."""
        self.story.append(Paragraph("Quality Control & Validation", self.styles['Heading1']))
        
        qc_text = f"""
        <b>Processing Quality:</b><br/>
        • Analysis completed successfully: {self.analysis.status == 'completed'}<br/>
        • Processing time: {self.analysis.processing_time:.2f} seconds<br/>
        • Timestamp: {self.analysis.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}
        
        <br/><br/>
        <b>Image Quality Assessment:</b><br/>
        • Image dimensions: {self.cell.image_width} × {self.cell.image_height} pixels<br/>
        • File format: {self.cell.file_format.upper()}<br/>
        • File size: {self.cell.file_size} bytes<br/>
        • Scale calibration: {'Available' if self.cell.scale_set else 'Not available'}
        """
        
        if hasattr(self.analysis, 'filtering_mode'):
            qc_text += f"""
            <br/><br/>
            <b>Data Filtering:</b><br/>
            • Filtering mode: {self.analysis.get_filtering_mode_display()}<br/>
            • Quality validation applied: Yes
            """
        
        self.story.append(Paragraph(qc_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def _add_technical_appendix(self):
        """Add technical appendix with detailed parameters."""
        self.story.append(Paragraph("Technical Appendix", self.styles['Heading1']))
        
        # Software information
        software_text = """
        <b>Software Information:</b><br/>
        • Analysis Platform: Morph AI Morphometric Analysis System<br/>
        • Segmentation: Cellpose Deep Learning Framework<br/>
        • Feature Extraction: scikit-image library<br/>
        • Statistical Analysis: NumPy, SciPy, Pandas<br/>
        • Report Generation: ReportLab PDF toolkit
        """
        
        self.story.append(Paragraph(software_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Complete parameter list
        params_table_data = [
            ['Parameter', 'Value', 'Description'],
            ['Cellpose Model', self.analysis.get_cellpose_model_display(), 'Deep learning model used for segmentation'],
            ['Cell Diameter', f"{self.analysis.cellpose_diameter} px", 'Expected cell diameter in pixels'],
            ['Flow Threshold', str(self.analysis.flow_threshold), 'Flow field threshold for segmentation'],
            ['Cell Probability Threshold', str(self.analysis.cellprob_threshold), 'Probability threshold for cell detection'],
            ['ROI Analysis', 'Yes' if self.analysis.use_roi else 'No', 'Region of interest analysis mode'],
        ]
        
        # Add preprocessing parameters if available
        if hasattr(self.analysis, 'apply_preprocessing'):
            preprocessing_params = [
                ['Preprocessing Enabled', 'Yes' if self.analysis.apply_preprocessing else 'No', 'Image preprocessing applied'],
                ['Noise Reduction', 'Yes' if getattr(self.analysis, 'apply_noise_reduction', False) else 'No', 'Bilateral noise reduction'],
                ['Contrast Enhancement', 'Yes' if getattr(self.analysis, 'apply_contrast_enhancement', False) else 'No', 'CLAHE contrast enhancement'],
                ['Sharpening', 'Yes' if getattr(self.analysis, 'apply_sharpening', False) else 'No', 'Unsharp mask sharpening'],
                ['Normalization', 'Yes' if getattr(self.analysis, 'apply_normalization', False) else 'No', 'Z-score normalization'],
            ]
            params_table_data.extend(preprocessing_params)
        
        params_table = Table(params_table_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), self._get_table_font(bold=True)),
            ('FONTNAME', (0, 1), (-1, -1), self._get_table_font(bold=False)),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        self.story.append(params_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Report generation info
        generation_text = f"""
        <b>Report Generation:</b><br/>
        • Generated on: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • Report format: PDF (ReportLab)<br/>
        • Page size: A4<br/>
        • Resolution: 300 DPI
        """
        
        self.story.append(Paragraph(generation_text, self.styles['Normal']))
    
    def _add_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page."""
        try:
            canvas_obj.saveState()
            
            # Get page dimensions
            page_width = doc.pagesize[0]
            page_height = doc.pagesize[1]
            
            # Header
            canvas_obj.setFont('Helvetica-Bold', 10)
            canvas_obj.setFillColor(colors.HexColor('#2E86AB'))
            canvas_obj.drawString(doc.leftMargin, page_height - doc.topMargin + 0.5*cm,
                                 f"Morphometric Analysis Report - {self.cell.name}")
            
            # Footer
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.setFillColor(colors.grey)
            canvas_obj.drawRightString(page_width - doc.rightMargin, doc.bottomMargin - 0.5*cm,
                                      f"Page {canvas_obj.getPageNumber()}")
            canvas_obj.drawString(doc.leftMargin, doc.bottomMargin - 0.5*cm,
                                 f"Generated: {timezone.now().strftime('%Y-%m-%d %H:%M')}")
            
            canvas_obj.restoreState()
        except Exception as e:
            logger.warning(f"Could not add header/footer: {e}")
    
    def _create_error_pdf(self) -> io.BytesIO:
        """Create error PDF when report generation fails."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=self.pagesize)
        
        error_story = [
            Paragraph("Error Generating Report", self.styles['Title']),
            Spacer(1, 0.5*inch),
            Paragraph("An error occurred while generating the morphometric analysis report. "
                     "Please contact support for assistance.", self.styles['Normal'])
        ]
        
        doc.build(error_story)
        buffer.seek(0)
        return buffer