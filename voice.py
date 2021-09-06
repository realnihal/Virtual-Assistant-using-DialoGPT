import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
import winsound
frequency = 440  # Set Frequency To 2500 Hertz
duration = 200  # Set Duration To 1000 ms == 1 second



print("Initialising the Model 1/2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
print("Initialising the Model 2/2")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
print("Initialising done!")

r = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

def takecommand():
    with sr.Microphone() as source:
        print("Listening!")
        audio = r.listen(source)
        print("Audio Captured")

    try:
        command = r.recognize_google(audio)
        command = command.lower()
        if 'alexa' in command:
            command = command.replace('alexa', '')
        else:
            command = "false"
    except sr.UnknownValueError:
        print("Alexa could not understand audio")
        command = "nothing"
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        command = "nothing"
    return command

def run_alexa():
    command = takecommand()
    print(command)
    if command == "nothing":
        print("Could'nt hear you try again!")
    elif command == 'false':
        print("Say Alexa to start a command!")
    elif command == 'stop':
        exit()
    elif 'play' in command:
        song = command.replace('play','')
        talk("playing " + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'tell me about' in command:
        person = command.replace('tell me about', '')
        try:
            info = wikipedia.summary(person, 1)
            print(info)
            talk(info)
        except:
            print("wikipedia coud not process the query")
    elif 'joke' in command:
        joke = pyjokes.get_joke()
        print(joke)
        talk(joke)
    else:
        new_user_input_ids = tokenizer.encode(command + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(text)
        talk(text)

winsound.Beep(frequency, duration)
print("starting the assistant")
while True:
    run_alexa()