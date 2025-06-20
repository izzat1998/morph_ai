from django.contrib import admin
from django.utils.html import format_html
from .models import Cell, CellAnalysis, DetectedCell


@admin.register(Cell)
class CellAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'image_width', 'image_height', 'file_format', 'file_size', 'has_analysis', 'analysis_count', 'created_at']
    list_filter = ['file_format', 'has_analysis', 'created_at', 'user']
    search_fields = ['name', 'user__email']
    readonly_fields = ['file_size', 'image_width', 'image_height', 'file_format', 'has_analysis', 'analysis_count', 'created_at', 'modified_at']
    ordering = ['-created_at']
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # editing an existing object
            return self.readonly_fields + ['image']
        return self.readonly_fields


class DetectedCellInline(admin.TabularInline):
    model = DetectedCell
    extra = 0
    readonly_fields = ['cell_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'extent', 
                      'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'centroid_x', 'centroid_y',
                      'bounding_box_x', 'bounding_box_y', 'bounding_box_width', 'bounding_box_height']
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(CellAnalysis)
class CellAnalysisAdmin(admin.ModelAdmin):
    list_display = ['id', 'cell_name', 'user', 'status', 'cellpose_model', 'num_cells_detected', 'processing_time', 'analysis_date']
    list_filter = ['status', 'cellpose_model', 'analysis_date', 'cell__user']
    search_fields = ['cell__name', 'cell__user__email']
    readonly_fields = ['num_cells_detected', 'processing_time', 'analysis_date', 'completed_at', 'error_message']
    ordering = ['-analysis_date']
    inlines = [DetectedCellInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('cell', 'status')
        }),
        ('Analysis Parameters', {
            'fields': ('cellpose_model', 'cellpose_diameter', 'flow_threshold', 'cellprob_threshold')
        }),
        ('Results', {
            'fields': ('segmentation_image', 'num_cells_detected', 'processing_time'),
            'classes': ('collapse',)
        }),
        ('Status & Timestamps', {
            'fields': ('analysis_date', 'completed_at', 'error_message'),
            'classes': ('collapse',)
        }),
    )
    
    def cell_name(self, obj):
        return obj.cell.name
    cell_name.short_description = 'Cell Name'
    
    def user(self, obj):
        return obj.cell.user.email
    user.short_description = 'User'
    
    def get_readonly_fields(self, request, obj=None):
        if obj and obj.status in ['completed', 'failed']:
            # If analysis is completed or failed, make most fields readonly
            return self.readonly_fields + ['cell', 'cellpose_model', 'cellpose_diameter', 
                                         'flow_threshold', 'cellprob_threshold']
        return self.readonly_fields


@admin.register(DetectedCell)
class DetectedCellAdmin(admin.ModelAdmin):
    list_display = ['analysis', 'cell_id', 'area', 'perimeter', 'circularity', 'eccentricity']
    list_filter = ['analysis__status', 'analysis__cell__user']
    search_fields = ['analysis__cell__name', 'analysis__cell__user__email']
    readonly_fields = ['analysis', 'cell_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 
                      'extent', 'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'centroid_x', 
                      'centroid_y', 'bounding_box_x', 'bounding_box_y', 'bounding_box_width', 'bounding_box_height']
    ordering = ['analysis', 'cell_id']
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False
