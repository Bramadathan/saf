from . import views
from django.urls import path

urlpatterns = [
    
    path('',views.index,name='index'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
]
