from gtts import gTTS

def text_to_sound(text, filename):
    # Tạo đối tượng gTTS
    tts = gTTS(text=text, lang='en')
    # Lưu vào file
    tts.save(filename)

text_to_sound('A person was detected', r'sound_alert\person_alert.mp3')
text_to_sound('A bicycle was detected', r'sound_alert\bicycle_alert.mp3')
text_to_sound('A car was detected', r'sound_alert\car_alert.mp3')