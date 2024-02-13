import gc
import json
import sys
from whisperspeech.pipeline import Pipeline
import io
import os
import re
import torch
import torchaudio
from pathlib import Path
from rvc_pipe.rvc_infer import rvc_convert

pipe = Pipeline(torch_compile=False, optimize=True)
use_rvc = True
text_examples = [
    [
        "This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer.",
        None,
    ],
    [
        "World War II or the Second World War was a global conflict that lasted from 1939 to 1945. The vast majority of the world's countries, including all the great powers, fought as part of two opposing military alliances: the Allies and the Axis.",
        "https://upload.wikimedia.org/wikipedia/commons/7/75/Winston_Churchill_-_Be_Ye_Men_of_Valour.ogg",
    ],
    [
        "<pl>To jest pierwszy test wielojęzycznego <en>Whisper Speech <pl>, modelu zamieniającego tekst na mowę, który Collabora i Laion nauczyli na superkomputerze <en>Jewels.",
        None,
    ],
    [
        '<en> WhisperSpeech is an Open Source library that helps you convert text to speech. <pl>Teraz także po Polsku! <en>I think I just tried saying "now also in Polish", don\'t judge me...',
        None,
    ],
    # ["<de> WhisperSpeech is multi-lingual <es> y puede cambiar de idioma <hi> मध्य वाक्य में"],
    ["<pl>To jest pierwszy test naszego modelu. Pozdrawiamy serdecznie.", None],
    # ["<en> The big difference between Europe <fr> et les Etats Unis <pl> jest to, że mamy tak wiele języków <uk> тут, в Європі"]
]


def parse_multilingual_text(input_text):
    pattern = r"(?:<(\w+)>)|([^<]+)"
    cur_lang = "en"
    segments = []
    for i, (lang, txt) in enumerate(re.findall(pattern, input_text)):
        if lang:
            cur_lang = lang
        else:
            segments.append(
                (cur_lang, f"  {txt}  ")
            )  # add spaces to give it some time to switch languages
    if not segments:
        return [("en", "")]
    return segments


def generate_audio(pipe, segments, speaker, speaker_url, cps=14):
    if isinstance(speaker, (str, Path)):
        speaker = pipe.extract_spk_emb(speaker)
    elif speaker_url:
        speaker = pipe.extract_spk_emb(speaker_url)
    else:
        speaker = pipe.default_speaker
    langs, texts = [list(x) for x in zip(*segments)]
    print(texts, langs)
    stoks = pipe.t2s.generate(texts, cps=cps, lang=langs)[0]
    atoks = pipe.s2a.generate(stoks, speaker.unsqueeze(0))
    audio = pipe.vocoder.decode(atoks)
    return audio.cpu()


def whisper_speech_demo(multilingual_text, speaker_audio=None, speaker_url="", cps=14):
    if len(multilingual_text) == 0:
        raise print("Please enter some text for me to speak!")

    segments = parse_multilingual_text(multilingual_text)

    audio = generate_audio(pipe, segments, speaker_audio, speaker_url, cps)

    return audio


# all rvc stuff

def do_gc():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        pass

RVC_SETTINGS = {
    'rvc_model': '',
    'f0_up_key': 0,
    'file_index': '',
	'index_rate' : 0,
    'filter_radius': 3,
    'resample_sr': 48000,
    'rms_mix_rate': 0.25,
    'protect': 0.33,
}

EXEC_SETTINGS = {}




#Stuff added by Jarod
def get_rvc_models():
	folder_path = 'models/rvc_models'
	return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pth')]
def get_rvc_indexes():
	folder_path = 'models/rvc_models'
	return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.index')]

def load_rvc_settings():
    global RVC_SETTINGS
    rvc_settings_path = './config/rvc.json'
    if os.path.exists(rvc_settings_path):
        with open(rvc_settings_path, 'r') as file:
            # RVC_SETTINGS.update(json.loads(file))
            return json.load(file)
    else:
        return {}  # Return an empty dict if the file doesn't exist



def update_rvc_settings(**kwargs):
    global RVC_SETTINGS
    RVC_SETTINGS.update(kwargs)
    save_rvc_settings()

def save_rvc_settings():
    global RVC_SETTINGS
    os.makedirs('./config/', exist_ok=True)
    with open(f'./config/rvc.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(RVC_SETTINGS, indent='\t'))




def run_rvc_infrence(output_voice):
    	#insert rvc stuff
	
		rvc_settings = load_rvc_settings()
		rvc_model_path = os.path.join("models", "rvc_models", rvc_settings['rvc_model'])
		rvc_index_path = os.path.join("models", "rvc_models", rvc_settings['file_index'])
		print (rvc_model_path)
		rvc_out_path = rvc_convert(model_path=rvc_model_path, 
							 		input_path=output_voice,
									f0_up_key=rvc_settings['f0_up_key'],
									file_index=rvc_index_path,
									index_rate=rvc_settings['index_rate'],
									filter_radius=rvc_settings['filter_radius'],
									resample_sr=rvc_settings['resample_sr'],
									rms_mix_rate=rvc_settings['rms_mix_rate'],
									protect=rvc_settings['protect'])
		
		# Read the contents from rvc_out_path
		with open(rvc_out_path, 'rb') as file:
			content = file.read()

		# Write the contents to output_voices[0], effectively replacing its contents
		with open(output_voice, 'wb') as file:
			file.write(content)

# outaudio = whisper_speech_demo("In moments when hope seems lost, remember, darkness is not eternal.", speaker_input, url_input)

# pipe = Pipeline(torch_compile=False,optimize=True)
# pipe.generate_to_file("output.wav", "In moments when hope seems lost, remember, darkness is not eternal. Like a candle in the night, hope illuminates the smallest corners of our lives. Nature teaches us, after the harshest storms come renewal and growth. Stand up, face the light, let it guide you out of despair. In unity, in love, in small acts of kindness, we find the strength to carry on. Hold on to hope, for the dawn is just beyond the horizon.")


# sys.path.append(os.path.join(now_dir,"/modules/rvc"))
# sys.path.append(os.path.join(now_dir,"/modules/uvr5"))
do_gc()

outaudio = whisper_speech_demo(
    "In moments when hope seems lost, remember, darkness is not eternal."
)

# # Export the output audio to the "out.wav" file
output_file_path = "out.wav"
sample_rate = 24000  # Assuming sample rate is 24000
torchaudio.save(output_file_path, outaudio, sample_rate)

if use_rvc:
     load_rvc_settings()
     run_rvc_infrence(output_file_path)
     


