from itertools import groupby

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC
from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2Attention,
                                                            Wav2Vec2Model)

import dlas.torch_intermediary as ml
from dlas.data.audio.unsupervised_audio_dataset import load_audio
from dlas.models.audio.tts.tacotron2.text import sequence_to_text
from dlas.trainer.networks import register_model
from dlas.utils.util import opt_get


def only_letters(string):
    allowlist = set(' ABCDEFGHIJKLMNOPQRSTUVWXYZ\'')
    return ''.join(filter(allowlist.__contains__, string.upper()))


class Wav2VecFeatureExtractor(nn.Module):
    """
    Basic wrapper that only does feature extraction. Useful to build out this portion of the model so it can be
    operated through DDP.
    """

    def __init__(self, basis_model='facebook/wav2vec2-large'):
        super().__init__()
        w2v = Wav2Vec2ForCTC.from_pretrained(basis_model)
        self.extractor = w2v.wav2vec2.feature_extractor

        for p in self.extractor.parameters():
            p.requires_grad = False
            p.DO_NOT_TRAIN = True

    def forward(self, audio, wav_lengths):
        with torch.no_grad():
            audio = audio[:, :, :wav_lengths.max()]
            audio_norm = (audio - audio.mean()) / \
                torch.sqrt(audio.var() + 1e-7)
            return self.extractor(audio_norm.squeeze(1))


class Wav2VecWrapper(nn.Module):
    """
    Basic wrapper class that makes Wav2Vec2 usable by DLAS.
    """

    def __init__(self, vocab_size=148, basis_model='facebook/wav2vec2-large', freeze_transformer=False, output_wer=True,
                 checkpointing_enabled=True, provide_attention_mask=False, spec_augment=True,
                 remove_feature_extractor=False, ramp_dropout_mode=False, ramp_dropout_end=20000, ramp_dropout_min=.1,
                 ramp_dropout_max=.5, layer_drop_pct=.1):
        super().__init__()
        self.provide_attention_mask = provide_attention_mask

        self.w2v = Wav2Vec2ForCTC.from_pretrained(basis_model)
        # Perform some surgery to get the model we actually want.
        self.w2v.wav2vec2.encoder.gradient_checkpointing = checkpointing_enabled
        self.w2v.lm_head = ml.Linear(self.w2v.config.hidden_size, vocab_size)
        self.w2v.config.vocab_size = vocab_size
        self.w2v.config.pad_token_id = 0
        self.w2v.config.ctc_loss_reduction = 'sum'
        self.w2v.config.apply_spec_augment = spec_augment
        self.w2v.config.layerdrop = layer_drop_pct
        self.remove_feature_extractor = remove_feature_extractor

        # This is a provision for distilling by ramping up dropout.
        self.ramp_dropout_mode = ramp_dropout_mode
        self.ramp_dropout_end = ramp_dropout_end
        self.ramp_dropout_min = ramp_dropout_min
        self.ramp_dropout_max = ramp_dropout_max
        self.current_dropout_rate = ramp_dropout_min

        if remove_feature_extractor:
            # The values passed in to the w2v model in this case are the outputs of the feature extractor.
            self.w2v.wav2vec2.feature_extractor = nn.Identity()
        else:
            # We always freeze the feature extractor, which needs some special operations in DLAS
            for p in self.w2v.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
                p.DO_NOT_TRAIN = True

        if freeze_transformer:
            # Also freeze the encoder here.
            for p in list(self.w2v.wav2vec2.encoder.parameters()) + list(self.w2v.wav2vec2.feature_projection.parameters()):
                p.requires_grad = False
                p.DO_NOT_TRAIN = True

        self.output_wer = output_wer
        if output_wer:
            self.last_pred = []
            self.last_labels = []

    def forward(self, audio, unaligned_tokens, wav_lengths, text_lengths, fea_extractor=None):
        unaligned_tokens = unaligned_tokens[:, :text_lengths.max()]
        audio = audio[:, :, :wav_lengths.max()]
        attention_mask = torch.ones_like(audio).squeeze(1)
        audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        audio = audio.squeeze(1)  # Get rid of the channels; w2v re-adds them.
        for b in range(audio.shape[0]):
            if self.provide_attention_mask:
                attention_mask[b, wav_lengths[b]:] = 0
            unaligned_tokens[b, text_lengths[b]:] = -100

        model_inp = fea_extractor if self.remove_feature_extractor else audio
        outputs = self.w2v(
            input_values=model_inp, attention_mask=attention_mask, labels=unaligned_tokens)

        if self.output_wer:
            self.last_pred.append(torch.argmax(outputs.logits, dim=-1))
            if len(self.last_pred) > 10:
                self.last_pred = self.last_pred[1:]
            self.last_labels.append(unaligned_tokens)
            if len(self.last_labels) > 10:
                self.last_labels = self.last_labels[1:]
        return outputs.loss

    def decode_ctc(self, output):
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        tokens = [token_group[0] for token_group in groupby(output)]
        filtered_tokens = list(filter(lambda token: token != 0, tokens))
        return filtered_tokens

    def get_debug_values(self, step, net_name):
        res = {}
        if self.output_wer and step % 100 == 0:
            from datasets import load_metric
            wer_metric = load_metric("wer")
            label_strings = []
            pred_strings = []
            for last_labels, last_pred in zip(self.last_labels, self.last_pred):
                last_labels[last_labels == -100] = 0
                label_strings.extend(
                    [only_letters(sequence_to_text(lbl)) for lbl in last_labels])
                pred_strings.extend([only_letters(sequence_to_text(
                    self.decode_ctc(pred))) for pred in last_pred])
            wer = wer_metric.compute(
                predictions=pred_strings, references=label_strings)
            res['wer'] = wer
            print(
                f"Sample prediction: {pred_strings[0]} <=> {label_strings[0]}")
        if self.ramp_dropout_mode:
            res['dropout_rate'] = self.current_dropout_rate
        return res

    def inference(self, audio):
        audio_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        logits = self.w2v(input_values=audio_norm.squeeze(1)).logits
        pred = logits.argmax(dim=-1)
        return [self.decode_ctc(p) for p in pred]

    def inference_logits(self, audio):
        audio_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        logits = self.w2v(input_values=audio_norm.squeeze(1)).logits
        return logits

    def update_for_step(self, step, *args):
        if self.ramp_dropout_mode and step % 10 == 0:
            dropout_gap = self.ramp_dropout_max - self.ramp_dropout_min
            new_dropout_rate = self.ramp_dropout_min + \
                dropout_gap * min(step / self.ramp_dropout_end, 1)
            self.current_dropout_rate = new_dropout_rate
            for name, module in self.w2v.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = new_dropout_rate
                elif isinstance(module, Wav2Vec2Attention):
                    module.dropout = new_dropout_rate


class Wav2VecBaseWrapper(nn.Module):
    def __init__(self, basis_model='facebook/wav2vec2-large'):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(basis_model)

    def forward(self, audio):
        audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        audio = audio.squeeze(1)  # Get rid of the channels; w2v re-adds them.
        outputs = self.w2v(input_values=audio)
        return outputs.last_hidden_state


@register_model
def register_wav2vec_feature_extractor(opt_net, opt):
    return Wav2VecFeatureExtractor(**opt_get(opt_net, ['kwargs'], {}))


@register_model
def register_wav2vec2_finetune(opt_net, opt):
    return Wav2VecWrapper(**opt_get(opt_net, ['kwargs'], {}))


@register_model
def register_wav2vec2(opt_net, opt):
    return Wav2VecBaseWrapper(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    fe = Wav2VecFeatureExtractor(basis_model='facebook/wav2vec2-large-960h')
    w2v = Wav2VecWrapper(basis_model='facebook/wav2vec2-large-960h',
                         freeze_transformer=True, remove_feature_extractor=True, ramp_dropout_mode=True)
    w2v.update_for_step(8000)
    fea = fe(torch.randn(2, 1, 50000), torch.tensor([20000, 30000]))
    loss = w2v(torch.randn(2, 1, 50000), torch.randint(0, 40, (2, 70)),
               torch.tensor([20000, 30000]), torch.tensor([35, 50]), fea)
    w2v.get_debug_values(0, "")

    sd = torch.load(
        '../experiments/train_wav2vec_mass_archived_r0/models/19500_wav2vec.pth')
    w2v.load_state_dict(sd)
    pred = w2v.inference(load_audio(
        'Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav', 16000).unsqueeze(0))
    res = sequence_to_text(pred[0])
    print(res)
