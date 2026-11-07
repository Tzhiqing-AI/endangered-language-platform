from django.contrib import admin

from .models import (
    Language, LanguageLocation, LanguageLocationDetail, TranslationMemory, Glossary,
    Region, Transmission, SpeakerNumberTrend, DomainOfUse, ScoreLevel, SourceType,
    Country, LanguageCode, LanguageSource, CountryRegion, LanguageCountry,
    BibliographyLink, LanguageVitality, SpeakerData, LanguageActivity,
    LanguageContext, LanguageOrthography, LanguageResource, LanguageMedia,
    ChangeSuggestion, Notification, NotificationUser, LanguageDocumentation,
    SampleCategory, LanguageSample, SampleCategoryLink, SampleContributorLink,
    SampleTranslation,
)


@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = ('language_id', 'language_name', 'level_of_endangerment',
                    'number_of_speakers', 'country', 'name_source')
    search_fields = ('language_name', 'country')
    list_filter = ('level_of_endangerment',)
    list_per_page = 50


@admin.register(LanguageCode)
class LanguageCodeAdmin(admin.ModelAdmin):
    list_display = ('id', 'iso_code', 'language_name', 'code_authorities')
    search_fields = ('iso_code', 'language_name')
    list_per_page = 100


@admin.register(LanguageSource)
class LanguageSourceAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'author', 'year', 'is_placeholder')
    search_fields = ('title', 'author')
    list_filter = ('is_placeholder',)
    list_per_page = 100


@admin.register(BibliographyLink)
class BibliographyLinkAdmin(admin.ModelAdmin):
    list_display = ('id', 'source_id', 'code_id', 'cited', 'documentation_index')
    list_per_page = 100


@admin.register(LanguageVitality)
class LanguageVitalityAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'endangerment_level',
                    'transmission_id', 'speaker_number_trends_id', 'preferred')
    list_filter = ('endangerment_level', 'preferred')
    list_per_page = 100


@admin.register(SpeakerData)
class SpeakerDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'speaker_number', 'ethnic_population', 'date_of_info')
    list_per_page = 100


@admin.register(LanguageActivity)
class LanguageActivityAdmin(admin.ModelAdmin):
    list_display = ('id', 'date', 'code_id', 'user_id', 'obj_type', 'obj_description')
    list_filter = ('obj_type',)
    list_per_page = 100


@admin.register(LanguageLocationDetail)
class LanguageLocationDetailAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'places', 'coordinates')
    list_per_page = 100


@admin.register(LanguageContext)
class LanguageContextAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'government_support', 'speaker_attitude')
    list_per_page = 100


@admin.register(LanguageOrthography)
class LanguageOrthographyAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'scripts', 'competing_orthographies')
    list_per_page = 100


@admin.register(LanguageResource)
class LanguageResourceAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'resource_type', 'url')
    list_per_page = 100


@admin.register(LanguageMedia)
class LanguageMediaAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'url', 'text')
    list_per_page = 100


@admin.register(ChangeSuggestion)
class ChangeSuggestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'object_id', 'reason_for_change', 'topic', 'user_id')
    list_per_page = 100


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'activity_id')
    list_per_page = 100


@admin.register(NotificationUser)
class NotificationUserAdmin(admin.ModelAdmin):
    list_display = ('id', 'notification_id', 'user_id')
    list_per_page = 100


@admin.register(LanguageDocumentation)
class LanguageDocumentationAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'grammar_score_id',
                    'dictionary_score_id', 'text_score_id')
    list_per_page = 100


@admin.register(LanguageSample)
class LanguageSampleAdmin(admin.ModelAdmin):
    list_display = ('id', 'code_id', 'title', 'sample_type', 'date', 'featured')
    list_filter = ('sample_type', 'featured')
    search_fields = ('title',)
    list_per_page = 100


@admin.register(SampleCategory)
class SampleCategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    list_per_page = 100


@admin.register(SampleCategoryLink)
class SampleCategoryLinkAdmin(admin.ModelAdmin):
    list_display = ('id', 'category_id', 'sample_id')
    list_per_page = 100


@admin.register(SampleContributorLink)
class SampleContributorLinkAdmin(admin.ModelAdmin):
    list_display = ('id', 'sample_id', 'contributor', 'role')
    list_per_page = 100


@admin.register(SampleTranslation)
class SampleTranslationAdmin(admin.ModelAdmin):
    list_display = ('id', 'sample_id', 'name', 'url')
    list_per_page = 100


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'iso_code')
    search_fields = ('name',)
    list_per_page = 100


@admin.register(CountryRegion)
class CountryRegionAdmin(admin.ModelAdmin):
    list_display = ('id', 'country_id', 'region_id')
    list_per_page = 100


@admin.register(LanguageCountry)
class LanguageCountryAdmin(admin.ModelAdmin):
    list_display = ('id', 'language_id', 'country_id')
    list_per_page = 100


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    list_per_page = 100


@admin.register(Transmission)
class TransmissionAdmin(admin.ModelAdmin):
    list_display = ('id', 'level', 'description')
    list_per_page = 100


@admin.register(SpeakerNumberTrend)
class SpeakerNumberTrendAdmin(admin.ModelAdmin):
    list_display = ('id', 'level', 'description')
    list_per_page = 100


@admin.register(DomainOfUse)
class DomainOfUseAdmin(admin.ModelAdmin):
    list_display = ('id', 'level', 'description')
    list_per_page = 100


@admin.register(ScoreLevel)
class ScoreLevelAdmin(admin.ModelAdmin):
    list_display = ('id', 'score_type', 'original_id', 'score', 'description')
    list_filter = ('score_type',)
    list_per_page = 100


@admin.register(SourceType)
class SourceTypeAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description')
    list_per_page = 100


@admin.register(LanguageLocation)
class LanguageLocationAdmin(admin.ModelAdmin):
    list_display = ('id', 'language_name', 'longitude', 'latitude')
    search_fields = ('language_name',)
    list_per_page = 100


admin.site.register(TranslationMemory)
admin.site.register(Glossary)
