from django import forms
from .models import Post, Reply, Report, Category, Tag


class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content', 'category', 'tags', 'related_language', 'language_level']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '请输入帖子标题...',
                'maxlength': 200
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请输入帖子内容...',
                'rows': 10
            }),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'tags': forms.SelectMultiple(attrs={'class': 'form-select'}),
            'related_language': forms.Select(attrs={'class': 'form-select'}),
            'language_level': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '如：极危、濒危、易危等'
            }),
        }


class ReplyForm(forms.ModelForm):
    class Meta:
        model = Reply
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请输入回复内容...',
                'rows': 3
            }),
        }


class ReportForm(forms.ModelForm):
    class Meta:
        model = Report
        fields = ['reason', 'description']
        widgets = {
            'reason': forms.Select(attrs={'class': 'form-select'}),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请详细描述举报原因...',
                'rows': 3
            }),
        }


class SearchForm(forms.Form):
    keyword = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '搜索帖子...'
        })
    )
    category = forms.ChoiceField(required=False, choices=[])
    tags = forms.ChoiceField(required=False, choices=[])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['category'].choices = [('', '全部分类')] + list(
            Category.objects.filter(is_active=True).values_list('id', 'name')
        )
        self.fields['tags'].choices = [('', '全部标签')] + list(
            Tag.objects.all().values_list('id', 'name')
        )
