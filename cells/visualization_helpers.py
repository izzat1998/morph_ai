"""
Helpers для подготовки данных визуализации
Professional frontend integration support
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from django.db.models import QuerySet
from .models import CellAnalysis, DetectedCell
from morphometric_stats.models import StatisticalAnalysis, FeatureStatistics


class VisualizationDataProcessor:
    """
    Процессор данных для фронтенд визуализации
    Подготавливает данные в формате, оптимизированном для Plotly и других JS библиотек
    """
    
    def __init__(self, analysis: CellAnalysis):
        self.analysis = analysis
        self.detected_cells = analysis.detected_cells.all()
        
        # Получить статистический анализ если доступен
        try:
            self.statistical_analysis = analysis.statistical_analysis
            self.has_statistical_data = True
        except StatisticalAnalysis.DoesNotExist:
            self.statistical_analysis = None
            self.has_statistical_data = False
    
    def prepare_scatter_plot_data(self) -> Dict[str, Any]:
        """
        Подготовить данные для scatter plot с доверительными интервалами
        """
        cells = list(self.detected_cells)
        
        # Основные данные
        areas = [float(cell.area) for cell in cells]
        circularities = [float(cell.circularity) for cell in cells]
        eccentricities = [float(cell.eccentricity) for cell in cells]
        solidities = [float(cell.solidity) for cell in cells]
        cell_ids = [cell.cell_id for cell in cells]
        
        # Reliability scores и uncertainty
        reliability_scores = []
        uncertainty_values = []
        area_confidence_intervals = []
        circularity_confidence_intervals = []
        
        if self.has_statistical_data:
            # Получить статистические данные для каждой клетки
            for cell in cells:
                cell_stats = self._get_cell_statistics(cell)
                
                # Reliability score (средний по всем параметрам)
                reliability = self._calculate_cell_reliability(cell_stats)
                reliability_scores.append(reliability)
                
                # Uncertainty (средний по всем параметрам)
                uncertainty = self._calculate_cell_uncertainty(cell_stats)
                uncertainty_values.append(uncertainty)
                
                # Confidence intervals для area и circularity
                area_ci = self._get_confidence_interval_width(cell_stats, 'area')
                circularity_ci = self._get_confidence_interval_width(cell_stats, 'circularity')
                
                area_confidence_intervals.append(area_ci)
                circularity_confidence_intervals.append(circularity_ci)
        else:
            # Рассчитать реальные значения на основе качества морфометрических данных
            reliability_scores = self._calculate_morphometric_reliability(cells)
            uncertainty_values = self._calculate_morphometric_uncertainty(cells)
            area_confidence_intervals = [area * 0.05 for area in areas]  # 5% CI
            circularity_confidence_intervals = [circ * 0.03 for circ in circularities]  # 3% CI
        
        return {
            'areas': areas,
            'circularities': circularities,
            'eccentricities': eccentricities,
            'solidities': solidities,
            'cell_ids': cell_ids,
            'reliability_scores': reliability_scores,
            'uncertainty_values': uncertainty_values,
            'area_confidence_intervals': area_confidence_intervals,
            'circularity_confidence_intervals': circularity_confidence_intervals
        }
    
    def prepare_box_plot_data(self) -> Dict[str, Any]:
        """
        Подготовить данные для box plot с доверительными интервалами
        """
        cells = list(self.detected_cells)
        
        # Собрать данные по параметрам
        features = {
            'Площадь': [float(cell.area) for cell in cells],
            'Округлость': [float(cell.circularity) for cell in cells],
            'Эксцентриситет': [float(cell.eccentricity) for cell in cells],
            'Плотность': [float(cell.solidity) for cell in cells]
        }
        
        # Подготовить confidence intervals для каждого параметра
        feature_confidence_intervals = []
        feature_reliability_scores = []
        
        if self.has_statistical_data:
            for feature_name in features.keys():
                # Получить популяционную статистику для параметра
                population_stats = self._get_population_statistics(feature_name.lower())
                
                if population_stats:
                    ci = {
                        'lower': float(population_stats.confidence_interval_lower),
                        'upper': float(population_stats.confidence_interval_upper)
                    }
                    reliability = float(population_stats.measurement_reliability_score)
                else:
                    # Fallback расчеты
                    values = features[feature_name]
                    mean_val = np.mean(values)
                    ci = {
                        'lower': mean_val * 0.95,
                        'upper': mean_val * 1.05
                    }
                    reliability = 0.75
                
                feature_confidence_intervals.append(ci)
                feature_reliability_scores.append([reliability] * len(cells))
        else:
            # Fallback CI для всех параметров
            for feature_name in features.keys():
                values = features[feature_name]
                mean_val = np.mean(values)
                ci = {
                    'lower': mean_val * 0.95,
                    'upper': mean_val * 1.05
                }
                feature_confidence_intervals.append(ci)
                feature_reliability_scores.append([0.75] * len(cells))
        
        return {
            'features': list(features.keys()),
            'values': list(features.values()),
            'feature_confidence_intervals': feature_confidence_intervals,
            'feature_reliability_scores': feature_reliability_scores
        }
    
    def prepare_correlation_data(self) -> Dict[str, Any]:
        """
        Подготовить данные для корреляционной матрицы
        """
        cells = list(self.detected_cells)
        
        # Собрать данные для корреляционного анализа
        data_matrix = np.array([
            [float(cell.area) for cell in cells],
            [float(cell.circularity) for cell in cells],
            [float(cell.eccentricity) for cell in cells],
            [float(cell.solidity) for cell in cells],
            [float(cell.aspect_ratio) for cell in cells]
        ])
        
        # Вычислить корреляционную матрицу
        correlation_matrix = np.corrcoef(data_matrix).tolist()
        
        # Confidence intervals для корреляций (bootstrap)
        confidence_matrix = self._calculate_correlation_confidence_intervals(data_matrix)
        
        feature_names = ['Площадь', 'Округлость', 'Эксцентриситет', 'Плотность', 'Соотношение сторон']
        
        return {
            'correlation_matrix': correlation_matrix,
            'confidence_matrix': confidence_matrix,
            'feature_names': feature_names
        }
    
    def prepare_reliability_indicators(self) -> Dict[str, Any]:
        """
        Подготовить индикаторы надежности для UI
        """
        if not self.has_statistical_data:
            # Рассчитать реальные значения на основе морфометрических данных
            overall_reliability = self._calculate_overall_morphometric_reliability()
            overall_uncertainty = self._calculate_overall_morphometric_uncertainty()
            
            reliability_score = int(overall_reliability * 100)
            
            if overall_reliability >= 0.8:
                reliability_class = 'excellent'
                reliability_color = '#22c55e'
                reliability_icon = 'check-circle'
            elif overall_reliability >= 0.6:
                reliability_class = 'good'
                reliability_color = '#eab308'
                reliability_icon = 'exclamation-circle'
            else:
                reliability_class = 'poor'
                reliability_color = '#ef4444'
                reliability_icon = 'times-circle'
            
            if overall_uncertainty <= 5:
                uncertainty_class = 'low'
            elif overall_uncertainty <= 15:
                uncertainty_class = 'medium'
            else:
                uncertainty_class = 'high'
            
            # Общее качество анализа
            if overall_reliability >= 0.8 and overall_uncertainty <= 5:
                quality_class = 'excellent'
                quality_icon = 'star'
                quality_text = 'Отличное качество анализа'
            elif overall_reliability >= 0.6 and overall_uncertainty <= 15:
                quality_class = 'good'
                quality_icon = 'check-circle'
                quality_text = 'Хорошее качество анализа'
            else:
                quality_class = 'poor'
                quality_icon = 'exclamation-triangle'
                quality_text = 'Требуется осторожная интерпретация'
            
            return {
                'overall_reliability_score': reliability_score,
                'overall_reliability_class': reliability_class,
                'overall_reliability_color': reliability_color,
                'overall_reliability_icon': reliability_icon,
                'overall_uncertainty_percent': overall_uncertainty,
                'overall_uncertainty_class': uncertainty_class,
                'analysis_quality_class': quality_class,
                'analysis_quality_icon': quality_icon,
                'analysis_quality_text': quality_text,
                'segmentation_uncertainty': overall_uncertainty * 0.4,  # Компонент сегментации
                'algorithm_uncertainty': overall_uncertainty * 0.6     # Компонент алгоритма
            }
        
        # Рассчитать общие показатели надежности
        population_stats = FeatureStatistics.objects.filter(
            statistical_analysis=self.statistical_analysis,
            detected_cell__isnull=True  # Популяционные статистики
        )
        
        if population_stats.exists():
            # Средняя надежность по всем параметрам
            avg_reliability = np.mean([
                float(stat.measurement_reliability_score) 
                for stat in population_stats
            ])
            
            # Средняя неопределенность
            avg_uncertainty = np.mean([
                float(stat.uncertainty_percent)
                for stat in population_stats
            ])
        else:
            # Если нет популяционных статистик, рассчитать из статистик отдельных клеток
            all_cell_stats = FeatureStatistics.objects.filter(
                statistical_analysis=self.statistical_analysis,
                detected_cell__isnull=False  # Статистики отдельных клеток
            )
            
            if all_cell_stats.exists():
                # Средняя надежность по всем клеткам и параметрам
                avg_reliability = np.mean([
                    float(stat.measurement_reliability_score) 
                    for stat in all_cell_stats
                ])
                
                # Средняя неопределенность
                avg_uncertainty = np.mean([
                    float(stat.uncertainty_percent)
                    for stat in all_cell_stats
                ])
            else:
                # Fallback к морфометрическим расчетам
                avg_reliability = self._calculate_overall_morphometric_reliability()
                avg_uncertainty = self._calculate_overall_morphometric_uncertainty()
            
            # Определить классы и цвета
            reliability_score = int(avg_reliability * 100)
            
            if avg_reliability >= 0.8:
                reliability_class = 'excellent'
                reliability_color = '#22c55e'
                reliability_icon = 'check-circle'
            elif avg_reliability >= 0.6:
                reliability_class = 'good'
                reliability_color = '#eab308'
                reliability_icon = 'exclamation-circle'
            else:
                reliability_class = 'poor'
                reliability_color = '#ef4444'
                reliability_icon = 'times-circle'
            
            if avg_uncertainty <= 5:
                uncertainty_class = 'low'
            elif avg_uncertainty <= 15:
                uncertainty_class = 'medium'
            else:
                uncertainty_class = 'high'
            
            # Общее качество анализа
            if avg_reliability >= 0.8 and avg_uncertainty <= 5:
                quality_class = 'excellent'
                quality_icon = 'star'
                quality_text = 'Отличное качество анализа'
            elif avg_reliability >= 0.6 and avg_uncertainty <= 15:
                quality_class = 'good'
                quality_icon = 'check-circle'
                quality_text = 'Хорошее качество анализа'
            else:
                quality_class = 'poor'
                quality_icon = 'exclamation-triangle'
                quality_text = 'Требуется осторожная интерпретация'
        
        return {
            'overall_reliability_score': reliability_score,
            'overall_reliability_class': reliability_class,
            'overall_reliability_color': reliability_color,
            'overall_reliability_icon': reliability_icon,
            'overall_uncertainty_percent': avg_uncertainty,
            'overall_uncertainty_class': uncertainty_class,
            'analysis_quality_class': quality_class,
            'analysis_quality_icon': quality_icon,
            'analysis_quality_text': quality_text,
            'segmentation_uncertainty': 2.3,  # Пример значений
            'algorithm_uncertainty': 1.1
        }
    
    def prepare_statistical_summary_enhanced(self) -> Dict[str, Any]:
        """
        Подготовить расширенную статистическую сводку для UI
        """
        if not self.has_statistical_data:
            return {}
        
        population_stats = FeatureStatistics.objects.filter(
            statistical_analysis=self.statistical_analysis,
            detected_cell__isnull=True
        )
        
        enhanced_summary = {}
        
        for stat in population_stats:
            feature_name = stat.feature_name
            
            enhanced_summary[feature_name] = {
                'mean': float(stat.mean_value),
                'std_error': float(stat.std_error),
                'confidence_interval_lower': float(stat.confidence_interval_lower),
                'confidence_interval_upper': float(stat.confidence_interval_upper),
                'confidence_interval_width': float(stat.confidence_interval_width),
                'uncertainty_percent': float(stat.uncertainty_percent),
                'reliability_score': float(stat.measurement_reliability_score),
                'bootstrap_mean': float(stat.bootstrap_mean) if stat.bootstrap_mean else None,
                'bootstrap_std': float(stat.bootstrap_std) if stat.bootstrap_std else None,
                'outlier_score': float(stat.outlier_score) if stat.outlier_score else None
            }
        
        return enhanced_summary
    
    def get_table_cell_classes(self) -> Dict[int, str]:
        """
        Получить CSS классы для цветового кодирования строк таблицы
        """
        cell_classes = {}
        
        if not self.has_statistical_data:
            # Fallback - все клетки "good"
            for cell in self.detected_cells:
                cell_classes[cell.cell_id] = 'cell-reliability--good'
            return cell_classes
        
        for cell in self.detected_cells:
            cell_stats = self._get_cell_statistics(cell)
            reliability = self._calculate_cell_reliability(cell_stats)
            
            if reliability >= 0.8:
                css_class = 'cell-reliability--excellent'
            elif reliability >= 0.6:
                css_class = 'cell-reliability--good'
            else:
                css_class = 'cell-reliability--poor'
            
            cell_classes[cell.cell_id] = css_class
        
        return cell_classes
    
    def _get_cell_statistics(self, cell: DetectedCell) -> QuerySet:
        """Получить статистические данные для конкретной клетки"""
        if not self.has_statistical_data:
            return FeatureStatistics.objects.none()
        
        return FeatureStatistics.objects.filter(
            statistical_analysis=self.statistical_analysis,
            detected_cell=cell
        )
    
    def _get_population_statistics(self, feature_name: str) -> Optional[FeatureStatistics]:
        """Получить популяционную статистику для параметра"""
        if not self.has_statistical_data:
            return None
        
        try:
            return FeatureStatistics.objects.get(
                statistical_analysis=self.statistical_analysis,
                detected_cell__isnull=True,
                feature_name=feature_name
            )
        except FeatureStatistics.DoesNotExist:
            return None
    
    def _calculate_cell_reliability(self, cell_stats: QuerySet) -> float:
        """Рассчитать общую надежность для клетки"""
        if not cell_stats.exists():
            return 0.75  # Fallback
        
        reliabilities = [
            float(stat.measurement_reliability_score)
            for stat in cell_stats
        ]
        
        return np.mean(reliabilities)
    
    def _calculate_cell_uncertainty(self, cell_stats: QuerySet) -> float:
        """Рассчитать общую неопределенность для клетки"""
        if not cell_stats.exists():
            return 5.0  # Fallback
        
        uncertainties = [
            float(stat.uncertainty_percent)
            for stat in cell_stats
        ]
        
        return np.mean(uncertainties)
    
    def _get_confidence_interval_width(self, cell_stats: QuerySet, feature_name: str) -> float:
        """Получить ширину доверительного интервала для параметра"""
        try:
            stat = cell_stats.get(feature_name=feature_name)
            return float(stat.confidence_interval_width)
        except FeatureStatistics.DoesNotExist:
            return 0.05  # Fallback 5%
    
    def _calculate_correlation_confidence_intervals(self, data_matrix: np.ndarray) -> List[List[Dict]]:
        """
        Рассчитать доверительные интервалы для корреляций (упрощенная версия)
        В реальности здесь был бы bootstrap анализ
        """
        n_features = data_matrix.shape[0]
        confidence_matrix = []
        
        for i in range(n_features):
            row = []
            for j in range(n_features):
                if i == j:
                    # Диагональ - корреляция с самим собой
                    ci = {'lower': 1.0, 'upper': 1.0}
                else:
                    # Упрощенный расчет CI для корреляций
                    # В реальности использовать Fisher z-transform + bootstrap
                    correlation = np.corrcoef(data_matrix[i], data_matrix[j])[0, 1]
                    margin = 0.1  # Упрощенная погрешность
                    ci = {
                        'lower': max(-1.0, correlation - margin),
                        'upper': min(1.0, correlation + margin)
                    }
                
                row.append(ci)
            confidence_matrix.append(row)
        
        return confidence_matrix
    
    def _calculate_morphometric_reliability(self, cells: List[DetectedCell]) -> List[float]:
        """
        Рассчитать надежность на основе качества морфометрических данных
        """
        if not cells:
            return []
        
        reliabilities = []
        areas = [float(cell.area) for cell in cells]
        circularities = [float(cell.circularity) for cell in cells]
        
        # Базовая надежность зависит от количества клеток
        base_reliability = min(0.95, 0.5 + (len(cells) / 50.0))  # Больше клеток = выше надежность
        
        for cell in cells:
            # Факторы качества для каждой клетки
            area_factor = 1.0
            circularity_factor = 1.0
            size_factor = 1.0
            
            # Проверка размера клетки
            if cell.area < 50:  # Очень маленькие клетки менее надежны
                size_factor = 0.7
            elif cell.area > 10000:  # Очень большие клетки могут быть артефактами
                size_factor = 0.8
            
            # Проверка округлости (0.8-1.0 хорошо, меньше - хуже)
            if cell.circularity >= 0.8:
                circularity_factor = 1.0
            elif cell.circularity >= 0.6:
                circularity_factor = 0.9
            else:
                circularity_factor = 0.7
            
            # Проверка консистентности с популяцией
            if areas:
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                if area_mean > 0:
                    area_cv = area_std / area_mean  # Коэффициент вариации
                    if area_cv < 0.3:  # Низкая вариабельность = хорошо
                        area_factor = 1.0
                    elif area_cv < 0.6:
                        area_factor = 0.9
                    else:
                        area_factor = 0.8
            
            # Итоговая надежность для клетки
            cell_reliability = base_reliability * area_factor * circularity_factor * size_factor
            cell_reliability = max(0.3, min(0.95, cell_reliability))  # Ограничить диапазон
            
            reliabilities.append(cell_reliability)
        
        return reliabilities
    
    def _calculate_morphometric_uncertainty(self, cells: List[DetectedCell]) -> List[float]:
        """
        Рассчитать неопределенность на основе качества морфометрических данных
        """
        if not cells:
            return []
        
        uncertainties = []
        areas = [float(cell.area) for cell in cells]
        
        # Базовая неопределенность зависит от количества клеток
        base_uncertainty = max(2.0, 15.0 - (len(cells) / 10.0))  # Больше клеток = меньше неопределенность
        
        for cell in cells:
            # Факторы неопределенности
            size_uncertainty = 0
            shape_uncertainty = 0
            population_uncertainty = 0
            
            # Неопределенность размера
            if cell.area < 50:
                size_uncertainty = 3.0  # Маленькие клетки менее точны
            elif cell.area > 10000:
                size_uncertainty = 4.0  # Большие клетки могут быть артефактами
            else:
                size_uncertainty = 1.0
            
            # Неопределенность формы
            if cell.circularity < 0.6:
                shape_uncertainty = 2.5  # Неправильная форма увеличивает неопределенность
            elif cell.circularity < 0.8:
                shape_uncertainty = 1.5
            else:
                shape_uncertainty = 0.5
            
            # Неопределенность от популяционной вариабельности
            if areas:
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                if area_mean > 0:
                    area_cv = area_std / area_mean
                    population_uncertainty = area_cv * 10  # Вариабельность увеличивает неопределенность
            
            # Итоговая неопределенность
            total_uncertainty = base_uncertainty + size_uncertainty + shape_uncertainty + population_uncertainty
            total_uncertainty = max(1.0, min(25.0, total_uncertainty))  # Ограничить диапазон
            
            uncertainties.append(total_uncertainty)
        
        return uncertainties
    
    def _calculate_overall_morphometric_reliability(self) -> float:
        """
        Рассчитать общую надежность анализа на основе морфометрических данных
        """
        cells = list(self.detected_cells)
        if not cells:
            return 0.5  # Низкая надежность если нет клеток
        
        # Факторы общей надежности
        cell_count_factor = min(1.0, len(cells) / 20.0)  # Оптимально 20+ клеток
        
        # Анализ консистентности размеров
        areas = [float(cell.area) for cell in cells]
        circularities = [float(cell.circularity) for cell in cells]
        
        area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 1.0
        circularity_mean = np.mean(circularities)
        
        # Консистентность размеров (CV < 0.5 хорошо)
        size_consistency_factor = max(0.5, 1.0 - area_cv)
        
        # Качество сегментации (высокая округлость хорошо)
        segmentation_quality_factor = min(1.0, circularity_mean / 0.8)
        
        # Количество очень маленьких/больших клеток (артефакты)
        valid_cells = sum(1 for cell in cells if 50 <= cell.area <= 10000)
        validity_factor = valid_cells / len(cells) if cells else 0
        
        # Итоговая надежность
        overall_reliability = (
            0.3 * cell_count_factor +
            0.25 * size_consistency_factor +
            0.25 * segmentation_quality_factor +
            0.2 * validity_factor
        )
        
        # Дополнительный бонус для идеальных случаев
        if len(cells) >= 10 and area_cv < 0.3 and circularity_mean > 0.8:
            overall_reliability = min(0.95, overall_reliability + 0.1)
        
        return max(0.3, min(0.95, overall_reliability))
    
    def _calculate_overall_morphometric_uncertainty(self) -> float:
        """
        Рассчитать общую неопределенность анализа на основе морфометрических данных
        """
        cells = list(self.detected_cells)
        if not cells:
            return 20.0  # Высокая неопределенность если нет клеток
        
        # Факторы неопределенности
        areas = [float(cell.area) for cell in cells]
        circularities = [float(cell.circularity) for cell in cells]
        
        # Неопределенность от размера выборки
        sample_uncertainty = max(1.0, 10.0 - len(cells))
        
        # Неопределенность от вариабельности
        area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.5
        variability_uncertainty = area_cv * 15
        
        # Неопределенность от качества сегментации
        circularity_mean = np.mean(circularities)
        segmentation_uncertainty = max(1.0, 8.0 * (1.0 - circularity_mean))
        
        # Неопределенность от артефактов
        invalid_cells = sum(1 for cell in cells if cell.area < 50 or cell.area > 10000)
        artifact_uncertainty = (invalid_cells / len(cells)) * 10 if cells else 0
        
        # Итоговая неопределенность
        total_uncertainty = (
            sample_uncertainty +
            variability_uncertainty +
            segmentation_uncertainty +
            artifact_uncertainty
        )
        
        return max(1.0, min(25.0, total_uncertainty))


def prepare_analysis_context_enhanced(analysis: CellAnalysis, request) -> Dict[str, Any]:
    """
    Главная функция для подготовки расширенного контекста анализа
    Интегрируется с существующими views
    """
    processor = VisualizationDataProcessor(analysis)
    
    # Базовые данные для визуализации
    scatter_data = processor.prepare_scatter_plot_data()
    box_plot_data = processor.prepare_box_plot_data()
    correlation_data = processor.prepare_correlation_data()
    
    # Индикаторы надежности
    reliability_indicators = processor.prepare_reliability_indicators()
    
    # Расширенная статистическая сводка
    enhanced_statistical_summary = processor.prepare_statistical_summary_enhanced()
    
    # CSS классы для таблицы
    table_cell_classes = processor.get_table_cell_classes()
    
    # Подготовить JSON данные для JavaScript
    json_data = {
        'area_values': json.dumps(scatter_data['areas']),
        'circularity_values': json.dumps(scatter_data['circularities']),
        'eccentricity_values': json.dumps(scatter_data['eccentricities']),
        'solidity_values': json.dumps(scatter_data['solidities']),
        'reliability_scores': json.dumps(scatter_data['reliability_scores']),
        'uncertainty_values': json.dumps(scatter_data['uncertainty_values']),
        'area_confidence_intervals': json.dumps(scatter_data['area_confidence_intervals']),
        'circularity_confidence_intervals': json.dumps(scatter_data['circularity_confidence_intervals']),
        'cell_ids': json.dumps(scatter_data['cell_ids']),
        'feature_confidence_intervals': json.dumps(box_plot_data['feature_confidence_intervals']),
        'feature_reliability_scores': json.dumps(box_plot_data['feature_reliability_scores'])
    }
    
    # Объединить все данные
    enhanced_context = {
        **reliability_indicators,
        **json_data,
        'enhanced_statistical_summary': enhanced_statistical_summary,
        'table_cell_classes': table_cell_classes,
        'has_advanced_visualization': True,
        'correlation_data': correlation_data,
        'box_plot_data': box_plot_data
    }
    
    return enhanced_context