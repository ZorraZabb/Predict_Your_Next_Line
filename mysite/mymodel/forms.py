from django import forms

class UserInputForm(forms.Form):
    user_input = forms.CharField(label='',widget=forms.TextInput(attrs={'placeholder': 'Give me some words'}), max_length=30)