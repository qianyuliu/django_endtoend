from django.db import models
from user.models import UserInfo

# Create your models here.


class PicTest(models.Model):
    test_pic = models.CharField(max_length=100)
    result_pic = models.CharField(max_length=100)
    name = models.CharField(max_length=20)
    age = models.IntegerField(default=0)
    gender = models.CharField(max_length=3)
    result = models.CharField(max_length=200)
    user = models.ForeignKey(UserInfo, on_delete=models.CASCADE)
