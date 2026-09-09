"""
Django 项目全局视图

包含错误处理函数等
"""
from django.shortcuts import render


def handler404(request, exception=None):
    """404 页面未找到错误处理"""
    return render(request, '404.html', status=404)


def handler500(request):
    """500 服务器内部错误处理"""
    return render(request, '500.html', status=500)


def index(request):
    """首页 - 液态玻璃风格入口页"""
    return render(request, 'index.html')
