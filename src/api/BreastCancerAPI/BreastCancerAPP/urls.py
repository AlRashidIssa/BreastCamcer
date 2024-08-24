from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_page, name='main'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('index/', views.index, name='index'),
    path('results/', views.result, name='results'),
]