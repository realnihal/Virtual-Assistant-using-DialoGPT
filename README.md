## Introduction
Ever wondered what it feels like to have your assistant. Someone that can do things that are personal to you. Luckily in modern times, we have many such services like "Alexa" and "Siri" to name a few. These services use your voice and collect data on you. To some people, this may not be fancy as it has privacy concerns. 

You can check out the linked [article here](https://realnihal.github.io/2021/09/07/Virtual-Assistant.html).

So let's work on making our **Virtual-Assistant**. First order of business, let's look at a couple of things that we want our assistant to do.

 1. Tell me the time
 2. Tell me a joke
 3. Tell me facts about anything I ask.
 4. Play a song on request
 
 **Most importantly!**
 
 5. **Have an utterly natural conversation (just like Alexa or Siri!)**

### Getting Started

The design I want to make involves the user "speaking out" the command to the computer (typing it is lame). 

We are going to use the [speech recognition](https://pypi.org/project/SpeechRecognition/) module in python to use. There are many backends that you can use with speech recognition. **Sphinx** is recommended for in device recommendation(more privacy). I will be using the Speech API of Google Cloud Platform(GCP).

```python
def takecommand():
	with sr.Microphone() as source:
		print("Listening!")
		audio = r.listen(source)
		print("Audio Captured")
	try:
		command = r.recognize_google(audio)
		command = command.lower()
		if 'jarvis' in command:
			command = command.replace('jarvis', '')
	except sr.UnknownValueError:
		print("Jarvis could not understand audio")
		command = "nothing"
	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))
		command = "nothing"
return command
````



>**And if you noticed, Yes I called my assistant JARVIS :)**

The return from the `recognise_google()` function is a text that is extracted from our audio. First, let's filter our command for the trigger word Jarvis. After that, we can remove the trigger word from the command and return the string.

Next, we are working on functionality. I have linked the resources below to useful python libraries, and you can always check out my code to see how they are implemented. They are pretty straightforward to figure out.

 - [Pyjokes](https://pypi.org/project/pyjokes/) - One-liner jokes for programmers (jokes as a service).
 - [Datetime](https://docs.python.org/3/library/datetime.html) - Basic Date and Time querying.
 - [Wikipedia](https://pypi.org/project/wikipedia/) - Module to query wiki articles.
 - [pywhatkit](https://pypi.org/project/pywhatkit/) - Can be used to do many things, including playing youtube videos.
 - [pyttsx3](https://pypi.org/project/pyttsx3/) - Text to Speech (TTS) library for Python Works without internet connection or delay. Supports multiple TTS engines, including Sapi5, nsss, and speak.

Our model can now speak, tell jokes, state facts, remind the time, and play music using these libraries.  You can add other features like weather or WhatsApp reminders; the world is at your fingertips. That's great now; let's start pushing the boundaries. We want to go beyond this. We want our model to converse like a human.

>We want our model to converse like a human.

### DialoGPT
Thanks to our friends at Microsoft, we have access to [DialoGPT](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/). DialoGPT adapts pretraining techniques to response generation **using hundreds of Gigabytes of colloquial data.** Like GPT-2, DialoGPT is formulated as an _autoregressive_ (AR) language model, and uses a multi-layer transformer as model architecture. Unlike GPT-2, which trains on general text data, DialoGPT draws on **147M multi-turn dialogues extracted from Reddit discussion threads**. Our implementation is based on the [huggingface pytorch-transformer](https://github.com/huggingface/transfer-learning-conv-ai) and [OpenAI GPT-2](https://github.com/openai/gpt-2).

   Please note that this model is highly resource-intensive and may lag your computer. Based on my testing, If you use Cuda compatible GPU to run python or have 4+ cores on your CPU, you should be okay. The following code has been added to the project to turn our assistant into an intelligent assistant. In other words now our model must be able to understand human speech and give apt responses to it.

```python
print("Initialising the Model 1/2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
print("Initialising the Model 2/2")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
print("Initialising done!")
new_user_input_ids = tokenizer.encode(command + tokenizer.eos_token, return_tensors='pt')
bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
print(text)
talk(text)
```
Now everything is ready, and it's time to test it! Let's see how our model performs and whether it's able to understand me.


![test case](https://github.com/realnihal/realnihal.github.io/blob/master/img/posts/assistant/test.jpg)

Great, this is amazing! Our assistant seems to be alive and can understand and converse with us. And it was lovely to hear her voice. This is a fantastic success!

You can feel free to check out my [article here](https://realnihal.github.io/2021/09/07/Virtual-Assistant.html).
