"""
URL configuration for language project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include,re_path
from django.shortcuts import render,HttpResponse
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from rest_framework.documentation import include_docs_urls
from rest_framework.permissions import AllowAny

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('admin/', admin.site.urls),
    path('Dashboard/',include('Dashboard.urls')),
    path('User/',include('User.urls')),
    path('Community/',include('Community.urls')),
    path('api/', include('rest_api.urls')),
    path('api/docs/', include_docs_urls(
        title='濒危语言保护平台 API',
        description='RESTful API 接口文档',
        permission_classes=[AllowAny],
        authentication_classes=[],
    )),
]

# 头像等媒体文件由 Django 提供（生产环境若上 CDN/nginx 可移除）
urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
else:
    # 无 nginx/whitenoise 的课程部署：collectstatic 后由 Django 提供静态文件
    urlpatterns += [
        re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
    ]

# 错误页面处理
handler404 = 'language.views.handler404'
handler500 = 'language.views.handler500'
