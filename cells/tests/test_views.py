from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
import io
import json

from cells.models import Cell, CellAnalysis

User = get_user_model()


class CellViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.client.login(email='test@example.com', password='testpass123')
        
        # Create test image
        self.test_image = self.create_test_image()
        
    def create_test_image(self):
        """Create a simple test image"""
        image = Image.new('RGB', (100, 100), color='white')
        image_io = io.BytesIO()
        image.save(image_io, format='PNG')
        image_io.seek(0)
        return SimpleUploadedFile(
            'test_cell.png',
            image_io.getvalue(),
            content_type='image/png'
        )
    
    def test_upload_cell_get(self):
        """Test GET request to upload cell page"""
        response = self.client.get(reverse('cells:upload'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'form')
    
    def test_upload_cell_post(self):
        """Test POST request to upload cell"""
        response = self.client.post(reverse('cells:upload'), {
            'name': 'Test Cell',
            'image': self.test_image,
            'description': 'Test description'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Cell.objects.filter(name='Test Cell').exists())
    
    def test_cell_list_view(self):
        """Test cell list view"""
        Cell.objects.create(
            name='Test Cell',
            user=self.user,
            image=self.test_image
        )
        
        response = self.client.get(reverse('cells:list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Cell')
    
    def test_cell_detail_view(self):
        """Test cell detail view"""
        cell = Cell.objects.create(
            name='Test Cell',
            user=self.user,
            image=self.test_image
        )
        
        response = self.client.get(reverse('cells:cell_detail', args=[cell.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Cell')
    
    def test_unauthorized_access(self):
        """Test that unauthorized users cannot access views"""
        self.client.logout()
        
        response = self.client.get(reverse('cells:upload'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        response = self.client.get(reverse('cells:list'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_analysis_list_view(self):
        """Test analysis list view"""
        response = self.client.get(reverse('cells:analysis_list'))
        self.assertEqual(response.status_code, 200)
    
    def test_analysis_status_ajax(self):
        """Test analysis status AJAX endpoint"""
        cell = Cell.objects.create(
            name='Test Cell',
            user=self.user,
            image=self.test_image
        )
        analysis = CellAnalysis.objects.create(
            cell=cell,
            cellpose_model='cyto',
            status='pending'
        )
        
        response = self.client.get(reverse('cells:analysis_status', args=[analysis.id]))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'pending')