from django import forms
from django.core.exceptions import ValidationError
from .models import UserProfile


class UserLoginForm(forms.Form):
    username = forms.CharField(min_length=1)
    password = forms.CharField(widget=forms.PasswordInput)


class UserProfileForm(forms.ModelForm):
    """用户资料表单"""
    name = forms.CharField(
        label='昵称',
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入昵称'})
    )

    class Meta:
        model = UserProfile
        fields = ['bio', 'phone', 'location', 'website']
        widgets = {
            'bio': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': '介绍一下自己吧'}),
            'phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入手机号'}),
            'location': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入所在地'}),
            'website': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'https://'}),
        }

    def __init__(self, *args, **kwargs):
        self.user_instance = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if self.user_instance:
            self.initial['name'] = self.user_instance.name

    def save(self, commit=True):
        profile = super().save(commit=commit)
        if self.user_instance and 'name' in self.cleaned_data:
            self.user_instance.name = self.cleaned_data['name']
            self.user_instance.save()
        return profile


class AvatarUploadForm(forms.ModelForm):
    """头像上传表单"""

    class Meta:
        model = UserProfile
        fields = ['avatar']
        widgets = {
            'avatar': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'avatar-input'
            })
        }

    def clean_avatar(self):
        avatar = self.cleaned_data.get('avatar')
        if avatar:
            if avatar.size > 2 * 1024 * 1024:
                raise ValidationError('头像图片不能超过 2MB')
            allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if avatar.content_type not in allowed_types:
                raise ValidationError('只支持 JPG、PNG、GIF、WebP 格式')
        return avatar
