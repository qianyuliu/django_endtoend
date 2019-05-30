from django.contrib import admin
from user.models import UserInfo
# Register your models here.

class UserInfoAdmin(admin.ModelAdmin):
    '''用户模型管理类'''
    list_display = ['id','username','password']

admin.site.register(UserInfo,UserInfoAdmin)
