from user import views
from django.urls import path

urlpatterns = [
    path('index/', views.index),
    path('login_check/', views.login_check),
    path('home/', views.home),
    path('logout/', views.logout),
    path('register/', views.register),
    path('detection/', views.detection),
    path('changepassword/',views.changepassword),
    path('updatepass/',views.updatepass)
]
