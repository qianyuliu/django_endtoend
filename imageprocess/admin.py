from django.contrib import admin
from imageprocess.models import PicTest


# Register your models here.

class PicTestAdmin(admin.ModelAdmin):
    '''用户模型管理类'''
    list_display = ['id', 'name', 'age', 'gender', 'result', 'test_pic', 'result_pic', 'user_id']


admin.site.register(PicTest, PicTestAdmin)
