from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _


class User(AbstractUser):
    email = models.EmailField(unique=True, verbose_name=_('Email'))
    first_name = models.CharField(max_length=30, verbose_name=_('First Name'))
    last_name = models.CharField(max_length=30, verbose_name=_('Last Name'))
    date_joined = models.DateTimeField(auto_now_add=True, verbose_name=_('Date Joined'))
    is_active = models.BooleanField(default=True, verbose_name=_('Active'))
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']
    
    def __str__(self):
        return self.email
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()
