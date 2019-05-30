from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from user.models import UserInfo


def index(request):
    if request.session.has_key('islogin'):
        return redirect('/home/')

    return render(request, 'user/index.html')


def login_check(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    users = UserInfo.objects.all()
    for user in users:
        if username == user.username and password == user.password:
            request.session['islogin'] = True
            request.session['username'] = username
            request.session['password'] = password

            request.session.set_expiry(0)
            return JsonResponse({'res': 1})

    return JsonResponse({'res': 0})


def register(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    users = UserInfo.objects.all()
    for user in users:
        if username == user.username:
            return JsonResponse({'res': 0})
    newuser = UserInfo()
    newuser.username = username
    newuser.password = password
    newuser.save()
    return JsonResponse({'res': 1})


def home(request):
    if request.session.has_key('islogin'):
        username = request.session.get('username')
        return render(request, 'user/home.html', {'username': username})
    else:
        return redirect('/index/')


def logout(request):
    request.session.flush()
    return redirect('/index/')


def detection(request):
    if request.session.has_key('islogin'):
        username = request.session.get('username')
        return render(request, 'user/detection.html', {'username': username})
    return redirect('/index/')


def changepassword(request):
    if request.session.has_key('islogin'):
        username = request.session.get('username')
        password = request.session.get('password')
        return render(request, 'user/changepassword.html', {'username': username, 'password': password})
    return redirect('/index/')


def updatepass(request):
    if request.session.has_key('islogin'):
        newpass = request.POST.get('newpass')
        oldpass = request.POST.get('oldpass')
        username = request.session.get('username')
        password = request.session.get('password')
        if oldpass == password:
            user = UserInfo.objects.get(username=username)
            if user:
                user.password = newpass
                user.save()
                return JsonResponse({'res': 1})
            return JsonResponse({'res': 2})
        return JsonResponse({'res': 0})
    return redirect('/index/')
