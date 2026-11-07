from django import forms
from Dashboard.models import Language

class LanguageForm(forms.ModelForm):
    class Meta:
        model = Language
        fields = [
            'language_name', 'legacy_endangerment_level', 'level_of_endangerment',
            'number_of_speakers', 'speaker_trends',
            'intergenerational_transmission', 'language_family', 'country',
            'additional_notes'
        ]
        widgets = {
            'additional_notes': forms.Textarea(attrs={'rows': 3}),
        }
