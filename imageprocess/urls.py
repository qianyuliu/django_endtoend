from imageprocess import views
from django.urls import path

urlpatterns = [
    path('uploadtest/', views.uploadtest),
    path('showlist/', views.showlist),
    path('savepic/',views.savepic),

]
