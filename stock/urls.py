from django.contrib import admin
from django.urls import path
from stock import views

urlpatterns = [
    path('', views.index,name="index"),
    path('index/', views.index,name="index"),
    path('about/', views.about,name="about"),
    path('contact/', views.contact,name="contact"),
    path('login/', views.login2,name="login"),
    path('logout/', views.logout2,name="logout"),
    path('pre/', views.pre,name="pre"),
    path('register/', views.register,name="pre"),
    path('forecast/', views.forecast,name="forecast"),
    path('profile/', views.profile,name="profile"),
]
