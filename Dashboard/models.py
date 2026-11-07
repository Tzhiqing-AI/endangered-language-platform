from django.db import models

from Dashboard.constants import LEGACY_LEVEL_LABELS, ELCAT_LEVEL_LABELS

# 代际传承 / 使用者趋势等级（源库 language_transmission / speaker_number_trends）
TRANSMISSION_LABELS = {
    '-1': '全龄使用',
    '1': '多数成人+部分儿童',
    '2': '多数成人，儿童少',
    '3': '部分成人，儿童不传',
    '4': '祖辈使用',
    '5': '仅少数老人',
    # 旧库 language 中的文本值
    'many': '多数人使用',
    'some': '部分人使用',
    'few': '少数人使用',
    'none': '几乎无人使用',
}

SPEAKER_TREND_LABELS = {
    '-1': '稳定/增长',
    '1': '缓慢下降',
    '2': '逐步下降',
    '3': '稳定下降',
    '4': '加速下降',
    '5': '快速下降',
    # 旧库 language 中的文本值
    'Increasing': '增长',
    'Stable': '稳定',
    'Decreasing': '下降',
}


# =============================================================================
# 兼容层：保持原结构，导入时回填 language_name / country
# 现有页面（language_views.py / ai_views.py / common.py）依赖这些字段
# =============================================================================

class Language(models.Model):
    """濒危语言模型"""
    language_id = models.AutoField(primary_key=True)
    language_name = models.CharField('语言名称', max_length=255, blank=True, null=True, db_index=True)
    level_of_endangerment = models.CharField('濒危等级', max_length=21, blank=True, null=True, db_index=True)
    legacy_endangerment_level = models.CharField(
        'UNESCO 五级分类', max_length=30, blank=True, null=True, db_index=True)
    number_of_speakers = models.IntegerField('使用者数量', blank=True, null=True)
    speaker_trends = models.CharField('使用者趋势', max_length=10, blank=True, null=True)
    intergenerational_transmission = models.CharField('代际传承', max_length=4, blank=True, null=True)
    language_family = models.CharField('语系', max_length=255, blank=True, null=True, db_index=True)
    country = models.CharField('国家', max_length=255, blank=True, null=True, db_index=True)
    additional_notes = models.TextField('附加说明', blank=True, null=True)
    media = models.CharField('媒体', max_length=255, blank=True, null=True)
    name_source = models.CharField('名称来源', max_length=500, blank=True, null=True)

    class Meta:
        db_table = 'language'
        verbose_name = '濒危语言'
        verbose_name_plural = '濒危语言'

    def __str__(self):
        return self.language_name or f'Language {self.language_id}'

    @property
    def legacy_level_display(self):
        return LEGACY_LEVEL_LABELS.get(self.legacy_endangerment_level, '未分类')

    @property
    def elcat_level_display(self):
        return ELCAT_LEVEL_LABELS.get(self.level_of_endangerment, '未分类')

    @property
    def transmission_display(self):
        return TRANSMISSION_LABELS.get(str(self.intergenerational_transmission), '未记录')

    @property
    def speaker_trend_display(self):
        return SPEAKER_TREND_LABELS.get(str(self.speaker_trends), '未记录')


class LanguageLocation(models.Model):
    """语言地理位置模型（兼容层，供地图使用）"""
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
    documentation = models.TextField('相关文档', blank=True, null=True)
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


# =============================================================================
# 查找表（等级 / 维度）——来自 language_region / transmission /
# speakernumbertrends / domainsofuse / *score / sourcetypes
# =============================================================================

class Region(models.Model):
    """大区查找表（源表 language_region）"""
    id = models.IntegerField(primary_key=True)
    name = models.CharField('大区名称', max_length=100, unique=True, default='')

    class Meta:
        db_table = 'region'
        verbose_name = '大区'
        verbose_name_plural = '大区'

    def __str__(self):
        return self.name


class Transmission(models.Model):
    """代际传承等级（源表 language_transmission，level -1~5）"""
    id = models.IntegerField(primary_key=True)
    level = models.IntegerField('等级值', default=0)
    description = models.TextField('描述', default='')

    class Meta:
        db_table = 'transmission'
        verbose_name = '代际传承等级'
        verbose_name_plural = '代际传承等级'

    def __str__(self):
        return f'{self.level}: {self.description[:30]}'


class SpeakerNumberTrend(models.Model):
    """使用者趋势等级（源表 language_speakernumbertrends，level -1~5）"""
    id = models.IntegerField(primary_key=True)
    level = models.IntegerField('等级值')
    description = models.TextField('描述')

    class Meta:
        db_table = 'speaker_number_trend'
        verbose_name = '使用者趋势等级'
        verbose_name_plural = '使用者趋势等级'

    def __str__(self):
        return f'{self.level}: {self.description[:30]}'


class DomainOfUse(models.Model):
    """使用领域等级（源表 language_domainsofuse，level -1~5）"""
    id = models.IntegerField(primary_key=True)
    level = models.IntegerField('等级值', default=0)
    description = models.TextField('描述', default='')

    class Meta:
        db_table = 'domain_of_use'
        verbose_name = '使用领域等级'
        verbose_name_plural = '使用领域等级'

    def __str__(self):
        return f'{self.level}: {self.description[:30]}'


class ScoreLevel(models.Model):
    """评分等级说明 dictionary/grammar/text score（4+5+5 条）

    三个评分表的原始 id 会重复（10/11/12/30），因此用自增主键，
    保留 original_id + score_type 组合唯一。
    """
    id = models.AutoField(primary_key=True)
    score_type = models.CharField('评分类型', max_length=20)  # dictionary/grammar/text
    original_id = models.IntegerField('原始ID')
    score = models.IntegerField('分值')
    description = models.TextField('说明', blank=True)

    class Meta:
        db_table = 'score_level'
        unique_together = ('score_type', 'original_id')
        verbose_name = '评分等级'
        verbose_name_plural = '评分等级'

    def __str__(self):
        return f'{self.score_type}-{self.original_id}: {self.score}'


class SourceType(models.Model):
    """来源类型（源表 language_sourcetypes）"""
    id = models.AutoField(primary_key=True)
    name = models.CharField('名称', max_length=100, unique=True)
    description = models.TextField('描述', blank=True)

    class Meta:
        db_table = 'source_type'
        verbose_name = '来源类型'
        verbose_name_plural = '来源类型'

    def __str__(self):
        return self.name


# =============================================================================
# 主数据表
# =============================================================================

class Country(models.Model):
    """国家（源表 language_country）"""
    id = models.AutoField(primary_key=True)
    name = models.CharField('国家名称', max_length=100, unique=True)
    iso_code = models.CharField('ISO代码', max_length=3, blank=True, null=True)

    class Meta:
        db_table = 'country'
        verbose_name = '国家'
        verbose_name_plural = '国家'

    def __str__(self):
        return self.name


class LanguageCode(models.Model):
    """语言代码（源表 language_codes）"""
    id = models.IntegerField(primary_key=True)
    iso_code = models.CharField('ISO代码', max_length=150, blank=True, null=True, db_index=True)
    code_authorities = models.CharField('代码机构', max_length=50, blank=True, null=True)
    language_name = models.CharField('语言名称', max_length=255, blank=True, null=True, db_index=True)
    alternate_names = models.TextField('别名', blank=True, null=True)
    classification = models.CharField('分类', max_length=500, blank=True, null=True)

    class Meta:
        db_table = 'language_code'
        verbose_name = '语言代码'
        verbose_name_plural = '语言代码'

    def __str__(self):
        return f'{self.iso_code or self.id} - {self.language_name or ""}'


class LanguageSource(models.Model):
    """文献主表（源表 language_sources）"""
    id = models.IntegerField(primary_key=True)
    source_type_id = models.IntegerField('来源类型ID', null=True, blank=True, db_index=True)
    address = models.CharField('地址', max_length=255, blank=True)
    author = models.CharField('作者', max_length=255, blank=True)
    booktitle = models.CharField('书名', max_length=255, blank=True)
    chapter = models.CharField('章节', max_length=100, blank=True)
    edition = models.CharField('版本', max_length=100, blank=True)
    editor = models.CharField('编者', max_length=255, blank=True)
    translator = models.CharField('译者', max_length=255, blank=True)
    howpublished = models.CharField('出版方式', max_length=255, blank=True)
    institution = models.CharField('机构', max_length=255, blank=True)
    journal = models.CharField('期刊', max_length=255, blank=True)
    month = models.CharField('月份', max_length=50, blank=True)
    note = models.CharField('备注', max_length=500, blank=True)
    num = models.CharField('编号', max_length=50, blank=True)
    pages = models.CharField('页码', max_length=50, blank=True)
    publisher = models.CharField('出版社', max_length=255, blank=True)
    school = models.CharField('学校', max_length=255, blank=True)
    series = models.CharField('系列', max_length=255, blank=True)
    title = models.CharField('标题', max_length=500, blank=True, db_index=True)
    url = models.CharField('链接', max_length=1000, blank=True)
    volume = models.CharField('卷', max_length=50, blank=True)
    year = models.IntegerField('年份', null=True, blank=True, db_index=True)
    isbn = models.CharField('ISBN', max_length=255, blank=True)
    free_text_citation = models.TextField('引用全文', blank=True)
    copied_from = models.CharField('复制来源', max_length=255, blank=True)
    is_placeholder = models.BooleanField('占位行', default=False, db_index=True)

    class Meta:
        db_table = 'language_source'
        verbose_name = '文献来源'
        verbose_name_plural = '文献来源'

    def __str__(self):
        return self.title or f'Source {self.id}'


# =============================================================================
# 关系表
# =============================================================================

class CountryRegion(models.Model):
    """国家-大区（源表 language_country_region）"""
    id = models.IntegerField(primary_key=True)
    country_id = models.IntegerField('国家ID', db_index=True)
    region_id = models.IntegerField('大区ID', db_index=True)

    class Meta:
        db_table = 'country_region'
        unique_together = ('country_id', 'region_id')
        verbose_name = '国家大区关系'
        verbose_name_plural = '国家大区关系'


class LanguageCountry(models.Model):
    """语言-国家多对多（源表 language_language_country）"""
    id = models.IntegerField(primary_key=True)
    language_id = models.IntegerField('语言ID', db_index=True)
    country_id = models.IntegerField('国家ID', db_index=True)

    class Meta:
        db_table = 'language_country_map'
        unique_together = ('language_id', 'country_id')
        verbose_name = '语言国家关系'
        verbose_name_plural = '语言国家关系'


class BibliographyLink(models.Model):
    """文献关联（源表 language_bibliography）"""
    id = models.IntegerField(primary_key=True)
    source_id = models.IntegerField('文献ID', db_index=True)
    code_id = models.IntegerField('语言ID', db_index=True)
    cited = models.BooleanField('被引用', default=False)
    documentation_index = models.IntegerField('文档索引', default=0)
    internal_status = models.CharField('内部状态', max_length=50, blank=True)

    class Meta:
        db_table = 'bibliography_link'
        verbose_name = '文献关联'
        verbose_name_plural = '文献关联'


# =============================================================================
# 语言明细表
# =============================================================================

class LanguageLocationDetail(models.Model):
    """位置明细（源表 language_locations，保留原始 ID）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True)
    places = models.CharField('地点', max_length=500, blank=True)
    coordinates = models.CharField('坐标', max_length=500, blank=True)
    description = models.TextField('描述', blank=True)
    public_comment = models.TextField('公开备注', blank=True)
    private_comment = models.CharField('内部备注', max_length=100, blank=True)
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)

    class Meta:
        db_table = 'language_location_detail'
        verbose_name = '位置明细'
        verbose_name_plural = '位置明细'


class LanguageVitality(models.Model):
    """语言活力评估（源表 language_vitality）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    endangerment_level = models.CharField('濒危等级', max_length=100, blank=True, db_index=True, default='')
    transmission_id = models.IntegerField('传承等级ID', null=True, blank=True)
    speaker_number_trends_id = models.IntegerField('趋势等级ID', null=True, blank=True)
    domains_of_use_id = models.IntegerField('领域等级ID', null=True, blank=True)
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.TextField('内部备注', blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)

    class Meta:
        db_table = 'language_vitality'
        verbose_name = '语言活力'
        verbose_name_plural = '语言活力'

    def __str__(self):
        return f'{self.code_id} - {self.endangerment_level}'


class SpeakerData(models.Model):
    """使用者统计（源表 language_speakers）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    speaker_number = models.IntegerField('使用者数', null=True, blank=True)
    speaker_number_text = models.TextField('使用者数原文', blank=True, default='')
    second_language_speakers = models.IntegerField('第二语言使用者', null=True, blank=True)
    semi_speakers = models.IntegerField('半使用者', null=True, blank=True)
    children = models.IntegerField('儿童', null=True, blank=True)
    young_adults = models.IntegerField('青年', null=True, blank=True)
    older_adults = models.IntegerField('老年', null=True, blank=True)
    elders = models.IntegerField('长老', null=True, blank=True)
    ethnic_population = models.IntegerField('族群人口', null=True, blank=True)
    date_of_info = models.CharField('数据日期', max_length=50, blank=True, default='')
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.TextField('内部备注', blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)

    class Meta:
        db_table = 'speaker_data'
        verbose_name = '使用者数据'
        verbose_name_plural = '使用者数据'

    def __str__(self):
        return f'{self.code_id} - {self.speaker_number}'


class LanguageActivity(models.Model):
    """动态/操作流（源表 language_activity）"""
    id = models.IntegerField(primary_key=True)
    date = models.DateTimeField('时间', null=True, blank=True, db_index=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    user_id = models.IntegerField('用户ID', null=True, blank=True, db_index=True)
    url = models.CharField('链接', max_length=500, blank=True, default='')
    obj_description = models.CharField('对象描述', max_length=500, blank=True, default='')
    obj_type = models.CharField('对象类型', max_length=50, blank=True, default='')
    level = models.IntegerField('等级', null=True, blank=True)
    type = models.IntegerField('类型', null=True, blank=True)
    obj_id = models.IntegerField('对象ID', null=True, blank=True)
    content_type_id = models.IntegerField('内容类型ID', null=True, blank=True)
    object_id = models.IntegerField('对象ID2', null=True, blank=True)

    class Meta:
        db_table = 'language_activity'
        verbose_name = '语言活动'
        verbose_name_plural = '语言活动'

    def __str__(self):
        return f'{self.code_id} - {self.obj_description[:30]}'


class LanguageContext(models.Model):
    """社会文化背景（源表 language_context）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    government_support = models.TextField('政府支持', blank=True, default='')
    institutional_support = models.TextField('机构支持', blank=True, default='')
    speaker_attitude = models.TextField('使用者态度', blank=True, default='')
    other_languages_used = models.CharField('其他语言', max_length=255, blank=True, default='')
    number_speaker_other_languages = models.CharField('其他语言使用者数', max_length=50, blank=True, default='')
    domains_other_langs = models.CharField('其他语言领域', max_length=255, blank=True, default='')
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.TextField('内部备注', blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)

    class Meta:
        db_table = 'language_context'
        verbose_name = '语言上下文'
        verbose_name_plural = '语言上下文'


class LanguageOrthography(models.Model):
    """正字法（源表 language_orthography）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    standard_orthography = models.TextField('标准正字法', blank=True, default='')
    scripts = models.CharField('书写系统', max_length=150, blank=True, default='')
    competing_orthographies = models.CharField('竞争正字法', max_length=150, blank=True, default='')
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.CharField('内部备注', max_length=100, blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)

    class Meta:
        db_table = 'language_orthography'
        verbose_name = '正字法'
        verbose_name_plural = '正字法'


class LanguageResource(models.Model):
    """资源（源表 language_resources，实际字段大多为空）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    resource_type = models.CharField('资源类型', max_length=100, blank=True, default='')
    description = models.TextField('描述', blank=True, default='')
    url = models.CharField('链接', max_length=500, blank=True, default='')
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.TextField('内部备注', blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)

    class Meta:
        db_table = 'language_resource'
        verbose_name = '语言资源'
        verbose_name_plural = '语言资源'


class LanguageMedia(models.Model):
    """媒体（源表 language_languagemedia）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    url = models.CharField('链接', max_length=500, default='')
    text = models.CharField('说明', max_length=50, blank=True, default='')

    class Meta:
        db_table = 'language_media'
        verbose_name = '语言媒体'
        verbose_name_plural = '语言媒体'


class ChangeSuggestion(models.Model):
    """变更建议（源表 language_changesuggestion）"""
    id = models.IntegerField(primary_key=True)
    content_type_id = models.IntegerField('内容类型ID', null=True, blank=True)
    object_id = models.IntegerField('对象ID', null=True, blank=True, db_index=True)
    reason_for_change = models.CharField('变更原因', max_length=200, blank=True, default='')
    topic = models.CharField('主题', max_length=200, blank=True, default='')
    explanation = models.TextField('说明', blank=True, default='')
    source = models.CharField('来源', max_length=1000, blank=True, default='')
    user_id = models.IntegerField('用户ID', null=True, blank=True)
    give_permission = models.BooleanField('授权', default=False)

    class Meta:
        db_table = 'change_suggestion'
        verbose_name = '变更建议'
        verbose_name_plural = '变更建议'


class Notification(models.Model):
    """通知（源表 language_notification）"""
    id = models.IntegerField(primary_key=True)
    activity_id = models.IntegerField('活动ID', db_index=True, default=0)

    class Meta:
        db_table = 'notification'
        verbose_name = '通知'
        verbose_name_plural = '通知'


class NotificationUser(models.Model):
    """通知接收人（源表 language_notification_users）"""
    id = models.IntegerField(primary_key=True)
    notification_id = models.IntegerField('通知ID', db_index=True, default=0)
    user_id = models.IntegerField('用户ID', db_index=True, default=0)

    class Meta:
        db_table = 'notification_user'
        verbose_name = '通知接收人'
        verbose_name_plural = '通知接收人'


class LanguageDocumentation(models.Model):
    """文档化记录（源表 language_documentation）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True, default=0)
    grammar_score_id = models.IntegerField('语法评分ID', null=True, blank=True)
    dictionary_score_id = models.IntegerField('词典评分ID', null=True, blank=True)
    text_score_id = models.IntegerField('文本评分ID', null=True, blank=True)
    public_comment = models.TextField('公开备注', blank=True, default='')
    private_comment = models.TextField('内部备注', blank=True, default='')
    source_id = models.IntegerField('来源ID', null=True, blank=True, db_index=True)
    preferred = models.BooleanField('首选', default=False)
    documentation_index = models.IntegerField('文档索引', default=0)

    class Meta:
        db_table = 'language_documentation'
        verbose_name = '文档化记录'
        verbose_name_plural = '文档化记录'


# =============================================================================
# 语言样本（声音、视频、词汇、语法、词典、图片）
# =============================================================================

class SampleCategory(models.Model):
    """样本分类 categories_category（8 条）"""
    id = models.IntegerField(primary_key=True)
    name = models.CharField('分类名称', max_length=100)
    description = models.TextField('说明', blank=True, default='')

    class Meta:
        db_table = 'sample_category'
        verbose_name = '样本分类'
        verbose_name_plural = '样本分类'

    def __str__(self):
        return self.name


class LanguageSample(models.Model):
    """语言样本 language_sample（6,805 条）"""
    id = models.IntegerField(primary_key=True)
    code_id = models.IntegerField('语言ID', db_index=True)
    url = models.CharField('链接', max_length=500, blank=True)
    title = models.CharField('标题', max_length=500, blank=True, db_index=True)
    recorded_by = models.CharField('记录者', max_length=255, blank=True)
    date = models.DateField('记录日期', null=True, blank=True)
    description = models.TextField('描述', blank=True)
    location = models.CharField('地点', max_length=500, blank=True)
    sample_type = models.CharField('类型', max_length=10, blank=True, db_index=True)
    user_id = models.IntegerField('用户ID', null=True, blank=True)
    original_publication = models.TextField('原始出版', blank=True)
    document_language = models.CharField('记录语言', max_length=100, blank=True)
    guide_type = models.CharField('指南类型', max_length=100, blank=True)
    rating_votes = models.IntegerField('评分人数', default=0)
    rating_score = models.FloatField('评分', null=True, blank=True)
    thumbnail_img = models.TextField('缩略图', blank=True)
    youtube_video_embed = models.TextField('视频嵌入', blank=True)
    contributor = models.CharField('贡献者', max_length=255, blank=True)
    coverage = models.TextField('覆盖范围', blank=True)
    creator = models.TextField('创作者', blank=True)
    publisher = models.TextField('出版者', blank=True)
    relation = models.TextField('关联', blank=True)
    rights = models.TextField('版权', blank=True)
    source = models.TextField('来源', blank=True)
    subject = models.TextField('主题', blank=True)
    view_count = models.IntegerField('浏览量', default=0)
    transcription = models.TextField('转写', blank=True)
    featured = models.BooleanField('精选', default=False)
    removed = models.BooleanField('已移除', default=False)
    added_at = models.DateTimeField('添加时间', null=True, blank=True)
    updated_at = models.DateTimeField('更新时间', null=True, blank=True)

    class Meta:
        db_table = 'language_sample'
        verbose_name = '语言样本'
        verbose_name_plural = '语言样本'
        ordering = ['-added_at', '-id']

    def __str__(self):
        return self.title or f'Sample {self.id}'


class SampleCategoryLink(models.Model):
    """样本-分类关联 language_samplecategory（14,779 条）"""
    id = models.IntegerField(primary_key=True)
    category_id = models.IntegerField('分类ID', db_index=True)
    sample_id = models.IntegerField('样本ID', db_index=True)

    class Meta:
        db_table = 'sample_category_link'
        verbose_name = '样本分类关联'
        verbose_name_plural = '样本分类关联'


class SampleContributorLink(models.Model):
    """样本-贡献者关联 language_samplecontributor（564 条）"""
    id = models.IntegerField(primary_key=True)
    sample_id = models.IntegerField('样本ID', db_index=True)
    contributor = models.CharField('贡献者', max_length=255, blank=True)
    role = models.CharField('角色', max_length=100, blank=True)

    class Meta:
        db_table = 'sample_contributor_link'
        verbose_name = '样本贡献者'
        verbose_name_plural = '样本贡献者'


class SampleTranslation(models.Model):
    """样本翻译/链接 language_sampletranslation（3 条）"""
    id = models.IntegerField(primary_key=True)
    sample_id = models.IntegerField('样本ID', db_index=True)
    name = models.CharField('名称', max_length=255, blank=True)
    url = models.CharField('链接', max_length=1000, blank=True)

    class Meta:
        db_table = 'sample_translation'
        verbose_name = '样本翻译'
        verbose_name_plural = '样本翻译'


# =============================================================================
# ETL 运行日志与数据归属（manage.py etl）
# =============================================================================

class ETLRun(models.Model):
    """一次 ETL 运行（rebuild / enrich / ml / pipeline）。"""

    STATUS_CHOICES = [
        ('running', '运行中'),
        ('success', '成功'),
        ('failed', '失败'),
        ('aborted', '已中止'),
    ]

    command = models.CharField('命令', max_length=50)
    source = models.CharField('数据源', max_length=20, blank=True, default='')
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='running')
    dry_run = models.BooleanField('演练模式', default=False)
    started_at = models.DateTimeField('开始时间', auto_now_add=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)
    backup_path = models.CharField('备份路径', max_length=500, blank=True, default='')
    options = models.TextField('运行参数', blank=True, default='')
    stats = models.TextField('统计结果', blank=True, default='')
    error = models.TextField('错误信息', blank=True, default='')

    class Meta:
        db_table = 'etl_run'
        verbose_name = 'ETL 运行'
        verbose_name_plural = 'ETL 运行'
        ordering = ['-started_at']

    def __str__(self):
        return f'{self.command} #{self.pk} {self.status}'


class ETLStep(models.Model):
    """ETL 运行中的阶段级日志，供 --resume 跳过已提交阶段。"""

    STATUS_CHOICES = [
        ('running', '运行中'),
        ('success', '成功'),
        ('failed', '失败'),
        ('skipped', '跳过'),
    ]

    run = models.ForeignKey(ETLRun, on_delete=models.CASCADE, related_name='steps')
    name = models.CharField('阶段', max_length=100)
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='running')
    started_at = models.DateTimeField('开始时间', auto_now_add=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)
    rows_affected = models.IntegerField('影响行数', default=0)
    message = models.TextField('说明', blank=True, default='')

    class Meta:
        db_table = 'etl_step'
        verbose_name = 'ETL 阶段'
        verbose_name_plural = 'ETL 阶段'
        unique_together = ('run', 'name')
        ordering = ['id']

    def __str__(self):
        return f'{self.name} {self.status}'


class ETLManagedRow(models.Model):
    """ETL 数据归属台账：只清理本系统写入且源库已不存在的行。

    以 (scope, table_name, row_id) 唯一标识一行；重建时按源库主键集合
    与台账求差，仅删除“曾由 ETL 写入、当前源库没有”的行，
    不会误删用户运行期产生的数据。
    """

    scope = models.CharField('归属域', max_length=32, default='elcat')
    table_name = models.CharField('目标表', max_length=64)
    row_id = models.BigIntegerField('目标主键')
    run_id = models.IntegerField('最近写入的 ETLRun', null=True, blank=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'etl_managed_row'
        verbose_name = 'ETL 行归属'
        verbose_name_plural = 'ETL 行归属'
        unique_together = ('scope', 'table_name', 'row_id')
        indexes = [models.Index(fields=['table_name'])]

    def __str__(self):
        return f'{self.scope}:{self.table_name}#{self.row_id}'
