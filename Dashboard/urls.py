from django.urls import path
from Dashboard.views import language, language_list, language_search, language_detail
from Dashboard.views import language_translation, language_visualization, language_map, language_chat
from Dashboard.views import get_chat_history
from Dashboard.views import nlp_page, nlp_process
from Dashboard.views import user_management, user_management_post
from Dashboard.views import translation_memory_list, translation_rate, glossary_manage
from Dashboard.views import admin_dashboard

app_name = 'Dashboard'

urlpatterns = [
    path('language/', language, name='language'),
    path('language/<int:language_id>/', language_detail, name='language_detail'),
    path('language_list/', language_list, name='language_list'),
    path('language_search/', language_search, name='language_search'),
    path('language_translation/', language_translation, name='language_translation'),
    path('language_visualization/', language_visualization, name='language_visualization'),
    path('language_map/', language_map, name='language_map'),
    path('language_chat/', language_chat, name='language_chat'),
    path('language_chat/history/', get_chat_history, name='language_chat_history'),
    path('nlp/', nlp_page, name='nlp_page'),
    path('nlp/process/', nlp_process, name='nlp_process'),
    path('user_management/', user_management, name='user_management'),
    path('user_management/post/', user_management_post, name='user_management_post'),
    path('translation_memory/', translation_memory_list, name='translation_memory_list'),
    path('translation/rate/', translation_rate, name='translation_rate'),
    path('glossary/', glossary_manage, name='glossary_manage'),
    path('admin_dashboard/', admin_dashboard, name='admin_dashboard'),
]
