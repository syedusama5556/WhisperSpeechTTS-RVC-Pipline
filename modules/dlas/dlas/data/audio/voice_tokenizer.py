import json
import re

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from dlas.data.audio.paired_voice_audio_dataset import (load_mozilla_cv,
                                                        load_tsv,
                                                        load_voxpopuli)
from dlas.models.audio.tts.tacotron2 import load_filepaths_and_text
from dlas.models.audio.tts.tacotron2.text.cleaners import english_cleaners


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(
        replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file, preprocess=None):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        self.language = vocab['model']['language'] if 'language' in vocab['model'] else None

        if preprocess is None:
            self.preprocess = 'pre_tokenizer' in vocab and vocab['pre_tokenizer']
        else:
            self.preprocess = preprocess

        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        if self.language == 'ja':
            import pykakasi

            kks = pykakasi.kakasi()
            results = kks.convert(txt)
            txt = " ".join([result['kana'] for result in results])
            txt = basic_cleaners(txt)
        else:
            txt = english_cleaners(txt)

        return txt

    def encode(self, txt):
        if self.preprocess:
            txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(
            seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')
        return txt


def build_text_file_from_priors(priors, output):
    with open(output, 'w', encoding='utf-8') as out:
        for p, fm in priors:
            if fm == 'lj' or fm == 'libritts':
                fetcher_fn = load_filepaths_and_text
            elif fm == 'tsv':
                fetcher_fn = load_tsv
            elif fm == 'mozilla_cv':
                fetcher_fn = load_mozilla_cv
            elif fm == 'voxpopuli':
                fetcher_fn = load_voxpopuli
            else:
                raise NotImplementedError()
            apt = fetcher_fn(p)
            for path, text in apt:
                out.write(text + "\n")
            out.flush()


def train():
    with open('all_texts.txt', 'r', encoding='utf-8') as at:
        ttsd = at.readlines()
    # bcd = datasets.load_dataset('bookcorpus', cache_dir='Z:\\huggingface_datasets\\cache')['train']

    # allowed_characters_re = re.compile(r'^[0-9a-z!@#%_=:;"/, \-\$\^&\*\(\)\+\{\[\]\}\\\.\'\?—–ʼ]+$')
    allowed_characters_re = re.compile(r'^[a-z!:;"/, \-\(\)\.\'\?ʼ]+$')

    def preprocess_word(word, report=False):
        word = english_cleaners(word)
        word = remove_extraneous_punctuation(word)
        if not bool(allowed_characters_re.match(word)):
            if report and word:
                print(f"REPORTING: '{word}'")
            return ''
        return word

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word(t, True) for t in ttsd[i:i+batch_size]]

        # print("Processing bookcorpus.")
        # for i in range(0, len(bcd), batch_size):
        #    yield [preprocess_word(t) for t in bcd[i:i+batch_size]['text']]

    trainer = BpeTrainer(
        special_tokens=['[STOP]', '[UNK]', '[SPACE]'], vocab_size=255)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        batch_iterator(), trainer, length=len(ttsd))  # +len(bcd))

    print(tokenizer.decode(tokenizer.encode(
        "i was traveling throughhadslfghds the woods in 1235375t137{{}}").ids))

    tokenizer.save('gpt_tts_tokenizer.json')


def test():
    tok = VoiceBpeTokenizer('gpt_tts_tokenizer.json')
    with open('all_texts.txt', 'r', encoding='utf-8') as at:
        ttsd = at.readlines()
        for line in ttsd:
            line = line.strip()
            seq = tok.encode(line)
            out = tok.decode(seq)
            print(f">>>{line}")
            print(f"<<<{out}")


if __name__ == '__main__':
    '''
    build_text_file_from_priors([('Y:\\bigasr_dataset\\libritts\\train-all.txt', 'libritts'),
                                 ('Y:\\bigasr_dataset\\libritts\\test-clean_list.txt', 'libritts'),
                                 #('Y:\\bigasr_dataset\\voxpopuli\\audio\\transcribed_data\\en\\asr_en.tsv', 'voxpopuli'),
                                 ('Y:\\bigasr_dataset\\voxpopuli\\audio\\transcribed_data\\en\\asr_train.tsv', 'voxpopuli'),
                                 ('Y:\\clips\\books1-transcribed.tsv', 'tsv'),
                                 ('Y:\\clips\\books2-transcribed.tsv', 'tsv'),
                                 ('Y:\\clips\\podcasts-0-transcribed.tsv', 'tsv')], 'all_texts.txt')
    '''
    # train()
    test()
