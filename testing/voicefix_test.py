import os
import sys
from voicefixer import VoiceFixer, Vocoder
from datetime import datetime
import git



def upscale_audio(input_file):


    current_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    # output_file_path = f"out_with_rvc_VOICEFIXER_{current_time}.wav"


    # TEST VOICEFIXER
    ## Initialize a voicefixer
    print("Initializing VoiceFixer...")
    voicefixer = VoiceFixer()
    # Mode 0: Original Model (suggested by default)
    # Mode 1: Add preprocessing module (remove higher frequency)
    # Mode 2: Train mode (might work sometimes on seriously degraded real speech)
    for mode in [0,1,2]:
        print("Testing mode",mode)
        output_file_path = f"{mode}_out_with_rvc_VOICEFIXER_{current_time}.wav"

        voicefixer.restore(input=os.path.join(os.getcwd(), input_file),  # Path to low quality .wav/.flac file
                        output=os.path.join(os.getcwd(), output_file_path),  # Path to save the file
                        cuda=False,  # Enable GPU acceleration
                        mode=mode)
    #     # if(mode != 2):
    #     #     check("output_mode_"+str(mode)+".flac")
    #     # print("Pass")

    # voicefixer.restore(input=os.path.join(os.getcwd(), input_file),  # Path to low quality .wav/.flac file
    #             output=os.path.join(os.getcwd(), output_file_path),  # Path to save the file
    #             cuda=False,  # Enable GPU acceleration
    #             mode=all)

    # TEST VOCODER
    ## Initialize a vocoder
    print("Initializing 44.1kHz speech vocoder...")
    vocoder = Vocoder(sample_rate=44100)

    ### read wave (fpath) -> mel spectrogram -> vocoder -> wave -> save wave (out_path)
    print("Test vocoder using groundtruth mel spectrogram...")
    output_file_path_2 = f"vocoder_out_with_rvc_VOICEFIXER_{current_time}.wav"

    vocoder.oracle(fpath=output_file_path,
                out_path=output_file_path_2,
                cuda=False) # GPU acceleration
 
    # sf.write(output_file_path, data=out_wav, samplerate=48000)

    print("Upscaler Done")

    return output_file_path


upscale_audio("C:\\Users\\Syed Usama Ahmad\\Documents\\All_AI_Projects_List\\WhisperSpeechRVCPipline\\results\\out_with_rvc_dropdown_13_02_2024__05_27_36.mp3")

     


