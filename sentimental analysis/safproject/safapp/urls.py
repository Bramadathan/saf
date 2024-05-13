from . import views
from django.urls import path

app_name = 'safapp'

urlpatterns = [
    
    path('',views.index,name='index'),
    path('register/', views.user_register, name='user_register'),
    path('login/', views.user_login, name='user_login'),
    path('logout/', views.user_logout, name='user_logout'),
]
