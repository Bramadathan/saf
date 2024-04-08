from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import SignupForm, LoginForm
def index(request):
    return render(request, 'index.html')

# signup page
def user_signup(request):
    if request.method == 'POST':
        # If the request method is POST, process the form data
        form = SignupForm(request.POST)
        if form.is_valid():
            # If the form data is valid, save the form
            form.save()
            # Redirect to the login page after successful signup
            return redirect('login')
    else:
        # If the request method is not POST, render the empty form
        form = SignupForm()
        # Render the signup form template with the form object
    return render(request, 'signup.html', {'form': form})



# login page
def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

# logout page
def user_logout(request):
    logout(request)
    return redirect('login')