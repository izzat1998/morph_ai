from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('analysis/<int:analysis_id>/pdf/', views.generate_analysis_pdf, name='analysis_pdf'),
    path('analysis/<int:analysis_id>/pdf/preview/', views.preview_analysis_pdf, name='analysis_pdf_preview'),
    path('analysis/<int:analysis_id>/pdf/config/', views.configure_analysis_pdf, name='analysis_pdf_config'),
]