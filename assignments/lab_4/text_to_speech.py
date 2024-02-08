from gtts import gTTS

speech_en = """This course introduces basic techniques for visualization, quantitative analysis, intelligent visual 
understanding, virtualization, digital animation, computer and video games, 
and web multimedia. Topics include data visualization, computer vision, visual analysis, 
the process of creating animated video clips from start to finish (including story 
creation, storyboarding, modeling, animation, and post-production), and computer virtualization; 
several key techniques include graphic design, video editing, motion generation, motion capture, 
multimedia, real-time rendering, visualization tools, and virtual machines. 
"""


tts_en = gTTS(speech_en, lang='en')
tts_en.save('waw/speech_en.mp3')

tts_fr = gTTS(speech_en, lang='fr')
tts_fr.save('waw/speech_fr.mp3')

tts_fr = gTTS(speech_en, lang='es')
tts_fr.save('waw/speech_es.mp3')

tts_fr = gTTS(speech_en, lang='ar')
tts_fr.save('waw/speech_ar.mp3')
