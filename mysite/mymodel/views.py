from django.shortcuts import render
from .forms import UserInputForm

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as sequence
import os

from PIL import Image, ImageDraw, ImageFont
from django.http import HttpResponse
from io import BytesIO
import base64
import textwrap

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'static', 'model_DIO.h5')
image_path = os.path.join(base_dir, 'static', 'images', 'your_nxt_lns.jpg')
data = """DIO: Tsugi wa Jotaro, kisama da!
    Jotaro: Yarou… DIO!
    DIO: Ho… mukatta kuruno ka? Nigetsu ni kono DIO ni chikazuite kuruno ka? Sekkaku sofu no Josefu ga watashi no Za Warudo no shotai wo. Shiken shuryu chaimu chokuzen made mondai yo toitte iru jukensee ne you na? Kisshi koita kibun de wo shietekure ta to yuu no ni?
    Jotaro: Chikadzu kanaka teme wo buchi no me tenain de na.
    DIO: Hoho! Dewa juubun chikazukanai youi.
    Jotaro: Ora!
    DIO: Noroi, noroi! Za Warudo wa saikyou no Sutando da. Jikan wa tomezetomo, supiido to paowa to te omae no Suta Purachina yoryuu enna no towa!
    Jotaro: Ore no Suta Purachina to onaji taipu wo Sutando nara. Enkyori enai kenai da, paowa to semitsu na bokina dekiru
    Jotaro and Dio Japanese Conversation-DIO: Tsugi wa Jotaro, kisama da!
    Jotaro: Yarou… DIO!
    DIO: Ho… mukatta kuruno ka? Niget
    """
model = keras.models.load_model(model_path)
tokenizer = Tokenizer()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])

def generate_text_image(text):
    margin = 60
    offset = 730
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('COOPBL.TTF', size=55)
    for line in textwrap.wrap(text, width=28):
        draw.text((margin, offset), line, font=font, fill=(255, 255, 255))
        offset += font.getsize(line)[1]
    # Convert the image to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64

# Create your views here.
def index(request):
    seed_text = ""
    if request.method == 'POST':
        form = UserInputForm(request.POST)
        if form.is_valid():
            seed_text = form.cleaned_data['user_input']
            next_word = 10
            for _ in range(next_word):
                # Generate text using your model
                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                token_list = sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') 
                predicted = np.argmax(model.predict(token_list), axis=-1)
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break
                seed_text += " " + output_word
            # Generate the image with the generated text in memory
            img_base64 = generate_text_image(seed_text)
    else:
        form = UserInputForm()
        img_base64 = None
    return render(request, 'index.html', {'form': form, 'prediction': seed_text, 'img_base64': img_base64})
    