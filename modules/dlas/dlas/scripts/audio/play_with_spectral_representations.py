import torchvision.utils

from dlas.utils.music_utils import music2cqt, music2mel
from dlas.utils.util import load_audio

if __name__ == '__main__':
    clip = load_audio('Y:\\split\\yt-music-eval\\00001.wav', 22050)
    mel = music2mel(clip)
    cqt = music2cqt(clip)
    torchvision.utils.save_image((mel.unsqueeze(1) + 1) / 2, 'mel.png')
    torchvision.utils.save_image((cqt.unsqueeze(1) + 1) / 2, 'cqt.png')
