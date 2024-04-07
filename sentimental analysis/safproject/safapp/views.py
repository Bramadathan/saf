from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import UserRegisterForm
from django.contrib import messages
import logging

logger = logging.getLogger(__name__)

def index(request):
    return render(request, "index.html")

def signup_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            messages.success(request, 'Your account has been created! You are now able to log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'signup.html', {'form': form, 'title': 'Register Here'})

def login_view(request):
    return render(request, "login.html")

