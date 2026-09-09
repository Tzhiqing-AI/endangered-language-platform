from django.db import models


# =============================================================================
# Dashboard 应用的模型 - 仅包含 Language 和 LanguageLocation
# User 和 ChatHistory 模型已移至 User 应用
# =============================================================================

class Language(models.Model):
    """濒危语言模型"""
    language_id = models.AutoField(primary_key=True)
    language_name = models.CharField('语言名称', max_length=255, blank=True, null=True, db_index=True)
    level_of_endangerment = models.CharField('濒危等级', max_length=21, blank=True, null=True, db_index=True)
    number_of_speakers = models.IntegerField('使用者数量', blank=True, null=True)
    speaker_trends = models.CharField('使用者趋势', max_length=10, blank=True, null=True)
    intergenerational_transmission = models.CharField('代际传承', max_length=4, blank=True, null=True)
    language_family = models.CharField('语系', max_length=255, blank=True, null=True, db_index=True)
    country = models.CharField('国家', max_length=255, blank=True, null=True, db_index=True)
    additional_notes = models.TextField('附加说明', blank=True, null=True)
    media = models.CharField('媒体', max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'language'
        verbose_name = '濒危语言'
        verbose_name_plural = '濒危语言'

    def __str__(self):
        return self.language_name or f'Language {self.language_id}'


class LanguageLocation(models.Model):
    """语言地理位置模型"""
    language_name = models.CharField('语言名称', max_length=255, blank=True, null=True)
    longitude = models.CharField('经度', max_length=255, blank=True, null=True)
    latitude = models.CharField('纬度', max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'language_location'
        verbose_name = '语言位置'
        verbose_name_plural = '语言位置'

    def __str__(self):
        return self.language_name or f'Location {self.id}'


class TranslationMemory(models.Model):
    """翻译记忆库 - 存储历史翻译结果"""
    id = models.AutoField(primary_key=True)
    source_text = models.TextField('原文', db_index=True)
    translated_text = models.TextField('译文')
    source_language = models.CharField('源语言', max_length=50, default='auto')
    target_language = models.CharField('目标语言', max_length=50, default='zh')
    username = models.CharField('翻译用户', max_length=150, blank=True, null=True)
    quality_score = models.IntegerField('质量评分', default=0, help_text='用户评分 1-5')
    usage_count = models.IntegerField('使用次数', default=1)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'translation_memory'
        verbose_name = '翻译记忆'
        verbose_name_plural = '翻译记忆'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.source_text[:30]} → {self.translated_text[:30]}'


class Glossary(models.Model):
    """术语表 - 存储专业术语对照"""
    id = models.AutoField(primary_key=True)
    term = models.CharField('术语', max_length=200, db_index=True)
    translation = models.CharField('翻译', max_length=200)
    language = models.CharField('所属语言', max_length=50, blank=True, null=True)
    category = models.CharField('分类', max_length=100, blank=True, null=True, help_text='如：语言学、文化、地理等')
    description = models.TextField('说明', blank=True, null=True)
    username = models.CharField('添加用户', max_length=150, blank=True, null=True)
    is_approved = models.BooleanField('已审核', default=False)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        db_table = 'glossary'
        verbose_name = '术语'
        verbose_name_plural = '术语表'
        ordering = ['term']

    def __str__(self):
        return f'{self.term} → {self.translation}'
