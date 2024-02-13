from datetime import datetime
import gc
import json
import random
import gradio as gr
import io
import os
import re
import torch
import torchaudio
from pathlib import Path
from whisperspeech.pipeline import Pipeline
from rvc_pipe.rvc_infer import rvc_convert
# from audiosr import build_model, super_resolution
# import numpy as np
# import soundfile as sf
from voicefixer import VoiceFixer, Vocoder

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

DEVEL = os.environ.get("DEVEL", False)

title = """

# Welcome to WhisperSpeech To RVC Pipline

WhisperSpeech is an Open Source text-to-speech system built by Collabora and LAION by inverting Whisper.
The model is fully open and you can run it on your local hardware. It's like **Stable Diffusion but for speech**
– both powerful and easily customizable.

Huge thanks to [Tonic](https://huggingface.co/Tonic) who helped build this Space for WhisperSpeech.

### How to Use It

Write you text in the box, you can use language tags (`<en>` or `<pl>`) to create multilingual speech.
Optionally you can upload a speech sample or give it a file URL to clone an existing voice. Check out the
examples at the bottom of the page for inspiration.
"""

footer = """

### Made By Syed Usama Ahmad

[https://github.com/syedusama5556](https://github.com/syedusama5556)

"""



#######################################################
# all rvc stuff

use_rvc = True
use_audio_upscaler = False
sample_rate = 24000  # Assuming sample rate is 24000

RVC_SETTINGS = {
    "rvc_model": "",
    "f0_up_key": 0,
    "file_index": "",
    "index_rate": 0.66,
    "filter_radius": 3,
    "resample_sr": 24000,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
}

EXEC_SETTINGS = {}


def do_gc():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        pass


def update_rvc_settings_proxy(*args):
    kwargs = {}
    keys = list(RVC_SETTINGS.keys())
    for i, key in enumerate(keys):
        kwargs[key] = args[i]

    update_rvc_settings(**kwargs)

def update_rvc_settings(**kwargs):
    global RVC_SETTINGS
    RVC_SETTINGS.update(kwargs)
    save_rvc_settings()

def save_rvc_settings():
    global RVC_SETTINGS
    print(RVC_SETTINGS)
    os.makedirs('./config/', exist_ok=True)
    with open(f'./config/rvc.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(RVC_SETTINGS, indent='\t'))


def on_rvc_model_change(selected_model, _):
    RVC_SETTINGS['rvc_model'] = selected_model
    save_rvc_settings()
    return selected_model

def on_index_rate_change(value, _):
    RVC_SETTINGS['index_rate'] = value
    save_rvc_settings()
    return value


# Stuff added by Jarod
def get_rvc_models():
    folder_path = "models/rvc_models"
    return [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".pth")
    ]


def get_rvc_indexes():
    folder_path = "models/rvc_models"
    return [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".index")
    ]


def load_rvc_settings():
    global RVC_SETTINGS
    rvc_settings_path = "./config/rvc.json"
    if os.path.exists(rvc_settings_path):
        try:
            with open(rvc_settings_path, "r") as file:
                RVC_SETTINGS.update(json.load(file))
                return RVC_SETTINGS
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {rvc_settings_path}: {e}")
            return {}
    else:
        print(f"File not found: {rvc_settings_path}")
        return {}

def run_rvc_infrence(output_voice):
    # insert rvc stuff

    rvc_settings = load_rvc_settings()
    rvc_model_path = os.path.join("models", "rvc_models", rvc_settings["rvc_model"])
    rvc_index_path = os.path.join("models", "rvc_models", rvc_settings["file_index"])
    print(rvc_model_path)
    print(RVC_SETTINGS)

    rvc_out_path = rvc_convert(
        model_path=rvc_model_path,
        input_path=output_voice,
        f0_up_key=rvc_settings["f0_up_key"],
        file_index=rvc_index_path,
        index_rate=rvc_settings["index_rate"],
        filter_radius=rvc_settings["filter_radius"],
        resample_sr=rvc_settings["resample_sr"],
        rms_mix_rate=rvc_settings["rms_mix_rate"],
        protect=rvc_settings["protect"],
    )

    # Read the contents from rvc_out_path
    with open(rvc_out_path, "rb") as file:
        content = file.read()

    # Write the contents to output_voices[0], effectively replacing its contents
    with open(output_voice, "wb") as file:
        file.write(content)


#######################################################
#AUDIO-SR

def upscale_audio(input_file):

   # Check if the "results" folder exists, if not, create it
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    current_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    output_file_path = f"{results_folder}/out_with_voicefixer_{current_time}.wav"


    # TEST VOICEFIXER
    ## Initialize a voicefixer
    print("Initializing VoiceFixer...")
    voicefixer = VoiceFixer()
    # Mode 0: Original Model (suggested by default)
    # Mode 1: Add preprocessing module (remove higher frequency)
    # Mode 2: Train mode (might work sometimes on seriously degraded real speech)
    # for mode in [0,1,2]:
    #     print("Testing mode",mode)
    #     voicefixer.restore(input=os.path.join(os.getcwd(), input_file),  # Path to low quality .wav/.flac file
    #                     output=os.path.join(os.getcwd(), output_file_path),  # Path to save the file
    #                     cuda=False,  # Enable GPU acceleration
    #                     mode=mode)

    voicefixer.restore(input= input_file,  # Path to low quality .wav/.flac file
                output= output_file_path,  # Path to save the file
                cuda=False,  # Enable GPU acceleration
                mode=all)
    ## Initialize a vocoder
    print("Initializing 44.1kHz speech vocoder...")
    vocoder = Vocoder(sample_rate=44100)

    ### read wave (fpath) -> mel spectrogram -> vocoder -> wave -> save wave (out_path)
    print("Test vocoder using groundtruth mel spectrogram...")
    vocoder.oracle(fpath=output_file_path,
                out_path=output_file_path,
                cuda=False) # GPU acceleration
 
    # sf.write(output_file_path, data=out_wav, samplerate=48000)

    print("Upscaler Done")

    return output_file_path

#####################################################


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


def whisper_speech_demo(
    multilingual_text, speaker_audio=None, speaker_url="", cps=14, use_rvc=False, use_audio_upscaler=False
):        
    global sample_rate
    do_gc()
    if len(multilingual_text) == 0:
        raise gr.Error("Please enter some text for me to speak!")

    segments = parse_multilingual_text(multilingual_text)

       # Check if the "results" folder exists, if not, create it
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    audio = generate_audio(pipe, segments, speaker_audio, speaker_url, cps)
    # Generate a unique random string
    current_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    
    # Construct the output file path with the unique string
    output_file_path = f"{results_folder}/out_with_rvc_{str(RVC_SETTINGS.get('rvc_model', ''))}_{current_time}.mp3"

    torchaudio.save(output_file_path, audio, sample_rate)

    if use_rvc:
        print("Running RVC")
        rvc_settings = list(RVC_SETTINGS.values())
        print(rvc_settings)
        run_rvc_infrence(output_file_path)
        audio, sample_rate_rvc = torchaudio.load(output_file_path)
        print("RVC Done")

    if use_audio_upscaler:
        output_file_path = upscale_audio(output_file_path)
        audio, sample_rate_rvc = torchaudio.load(output_file_path)
        print("upscale2 Done")

    return ((sample_rate_rvc if  (use_rvc or use_audio_upscaler) else sample_rate), audio.T.numpy())

    # Did not work for me in Safari:
    # mp3 = io.BytesIO()
    # torchaudio.save(mp3, audio, 24000, format='mp3')
    # return mp3.getvalue()


def update_voices():
    return (
        gr.Dropdown.update(choices=get_rvc_models()),  # Update for RVC models
        gr.Dropdown.update(choices=get_rvc_indexes()),  # Update for RVC models
    )


pipe = Pipeline(torch_compile=False, optimize=True)
# warmup will come from regenerating the examples
load_rvc_settings()

with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter multilingual text💬📝",
                value="World War II or the Second World War was a global conflict that lasted from 1939 to 1945.",
                info="You can use `<en>` for English and `<pl>` for Polish, see examples below.",
            )
            cps = gr.Slider(
                value=14,
                minimum=10,
                maximum=15,
                step=0.25,
                label="Tempo (in characters per second)",
            )

            with gr.Row():
                speaker_input = gr.Audio(
                    label="Upload or Record Speaker Audio (optional)🌬️💬",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                url_input = gr.Textbox(
                    label="alternatively, you can paste in an audio file URL:"
                )

            gr.Markdown("  \n  ")  # fixes the bottom overflow from Audio
            generate_button = gr.Button("Generate Speech🌟")

        with gr.Column(scale=1):
            output_audio = gr.Audio(label="WhisperSpeech says…")

    with gr.Column():
        use_audio_upscaler = gr.Checkbox(label="Upscale the outputted audio with voice fixer (beta)", value=False)
        use_rvc = gr.Checkbox(label="Run the outputted audio through RVC", value=True)
        with gr.Column(visible=use_rvc) as rvc_column:
            with gr.Row():
                refresh_voices = gr.Button(value="Refresh Voice List")
            RVC_SETTINGS["rvc_model"] = gr.Dropdown(
                choices=get_rvc_models(),
                label="RVC Voice Model",
                value=RVC_SETTINGS["rvc_model"],
                interactive=True,
            )
            RVC_SETTINGS["file_index"] = gr.Dropdown(
                choices=get_rvc_indexes(),
                label="RVC Index File",
                value=RVC_SETTINGS["file_index"],
                interactive=True,
            )
            RVC_SETTINGS["index_rate"] = gr.Slider(
                minimum=0,
                maximum=1,
                label="Index Rate",
                value=RVC_SETTINGS["index_rate"],
                interactive=True,
            )

            # RVC_SETTINGS["rvc_model"].change(fn=on_rvc_model_change, inputs=[RVC_SETTINGS["rvc_model"]], outputs=[RVC_SETTINGS["rvc_model"]])
            # RVC_SETTINGS["index_rate"].change(fn=on_index_rate_change, inputs=[RVC_SETTINGS["index_rate"]], outputs=[RVC_SETTINGS["index_rate"]])

            RVC_SETTINGS["f0_up_key"] = gr.Slider(
                minimum=-24,
                maximum=24,
                label="Pitch (No Change: 0, Male to Female: 12, Female to Male: -12):",
                value=RVC_SETTINGS["f0_up_key"],
                interactive=True,
            )
            # RVC_SETTINGS['f0_method'] = gr.Dropdown(choices=get_rvc_models(), label="RVC Voice Model", value=args.rvc_model)
            RVC_SETTINGS["filter_radius"] = gr.Slider(
                minimum=0,
                maximum=7,
                label="Filter Radius",
                value=RVC_SETTINGS["filter_radius"],
                interactive=True,
            )
            RVC_SETTINGS["resample_sr"] = gr.Slider(
                minimum=0,
                maximum=48000,
                label="Resample sample rate",
                value=RVC_SETTINGS["resample_sr"],
                interactive=True,
            )
            RVC_SETTINGS["rms_mix_rate"] = gr.Slider(
                minimum=0,
                maximum=1,
                label="RMS Mix Rate (Volume Envelope)",
                value=RVC_SETTINGS["rms_mix_rate"],
                interactive=True,
            )
            RVC_SETTINGS["protect"] = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="Protect Voiceless Consonants",
                value=RVC_SETTINGS["protect"],
                interactive=True,
            )

            rvc_inputs = list(RVC_SETTINGS.values())

            for k, component in RVC_SETTINGS.items():
                if isinstance(component, gr.Dropdown):
                    component.change(fn=update_rvc_settings_proxy, inputs=rvc_inputs)
                elif isinstance(component, gr.Slider):
                    component.release(fn=update_rvc_settings_proxy, inputs=rvc_inputs)



    generate_button.click(
        whisper_speech_demo,
        inputs=[text_input, speaker_input, url_input, cps,use_rvc,use_audio_upscaler],
        outputs=output_audio,
    )
    refresh_voices.click(
        update_voices,
        inputs=None,
        outputs=[
            RVC_SETTINGS["rvc_model"],  # Add this line
            RVC_SETTINGS["file_index"],
        ],
    )
    use_rvc.change(
        fn=lambda use_rvc_checked: gr.update(visible=use_rvc_checked),
        inputs=use_rvc,
        outputs=rvc_column,
    )
    gr.Markdown(footer)



demo.launch(
    server_port=3000,
    debug=True,
    enable_queue=True,
    server_name="0.0.0.0",
    share=False
)
