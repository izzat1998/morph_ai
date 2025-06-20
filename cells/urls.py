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
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('analysis/', views.analysis_list, name='analysis_list'),
    path('analysis/<int:analysis_id>/status/', views.analysis_status, name='analysis_status'),
    path('analysis/<int:analysis_id>/export/', views.export_analysis_csv, name='export_analysis_csv'),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
]