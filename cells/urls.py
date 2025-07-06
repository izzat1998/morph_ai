from django.urls import path
from . import views

app_name = 'cells'

urlpatterns = [
    # Cell management
    path('upload/', views.upload_cell, name='upload'),
    path('list/', views.cell_list, name='list'),
    path('<int:cell_id>/', views.cell_detail, name='cell_detail'),
    path('<int:cell_id>/set-scale/', views.set_scale_calibration, name='set_scale_calibration'),
    
    # Analysis functionality
    path('<int:cell_id>/analyze/', views.analyze_cell, name='analyze_cell'),
    path('<int:cell_id>/optimize-parameters/', views.optimize_analysis_parameters, name='optimize_parameters'),
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('analysis/<int:analysis_id>/enhanced/', views.enhanced_analysis_detail, name='enhanced_analysis_detail'),
    path('analysis/', views.analysis_list, name='analysis_list'),
    path('analysis/<int:analysis_id>/status/', views.analysis_status, name='analysis_status'),
    path('analysis/<int:analysis_id>/export/', views.export_analysis_csv, name='export_analysis_csv'),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
    
    # Batch processing (RTX 2070 Max-Q optimized)
    path('batch/', views.batch_list, name='batch_list'),
    path('batch/create/', views.batch_create, name='batch_create'),
    path('batch/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    path('batch/<int:batch_id>/configure/', views.batch_configure, name='batch_configure'),
    path('batch/<int:batch_id>/progress/', views.batch_progress, name='batch_progress'),
    path('batch/<int:batch_id>/cancel/', views.batch_cancel, name='batch_cancel'),
    path('batch/<int:batch_id>/delete/', views.batch_delete, name='batch_delete'),
    
    # Batch statistics JSON API endpoints
    path('batch/<int:batch_id>/statistics/json/', views.batch_statistics_json, name='batch_statistics_json'),
    path('batch/<int:batch_id>/visualization-data/', views.batch_visualization_data, name='batch_visualization_data'),
    path('batch/<int:batch_id>/feature-comparison/', views.batch_feature_comparison_json, name='batch_feature_comparison_json'),
]